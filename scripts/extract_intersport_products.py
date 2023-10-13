import itertools
import logging
import multiprocessing
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import attrs
import click
import joblib
import orjson
import polars as pl
import regex
from attr import asdict
from bs4 import BeautifulSoup
from icecream import ic
from loguru import logger
from selenium.webdriver.remote.webdriver import WebDriver
from tqdm import tqdm

from product_prediction.definitions import ProductInfo
from product_prediction.scraping import (
    MissingProductInfoException,
    get_headless_firefox_driver,
    url_to_soup,
)
from product_prediction.utils import flatten, split_into_n_chunks

GENDERS = ["heren", "dames"]
SHOE_CATEGORIES = [
    "hardloopschoenen",
    "voetbalschoenen",
    "wandelschoenen",
    "wielerschoenen",
    "hockeyschoenen",
    "tennisschoenen",
    "zaalschoenen",
]

INTERSPORT_SHOE_CATEGORY_BASE_URL = "https://www.intersport.nl/{}/schoenen/{}/"

ONLY_WHITE_SPACE = regex.compile(r"^\s*&")


@attrs.define
class ProductURLInfo:
    product_url: str
    category: str
    gender: str


def extract_intersport_product_urls(gender: str, category: str) -> List[str]:
    """Extract the products-urls given the url of the product-category page for the Intersport website (as of 10-10-2023)"""

    category_page_url = INTERSPORT_SHOE_CATEGORY_BASE_URL.format(gender, category)

    # By default load first 5 pages of product
    page = 5

    def mk_param_url(url_: str, page_: int):
        url_ = url_.split("?")[0]
        return f"{url_}?page={page_}"

    logger.debug(f"Extracting first pages for: {category_page_url}")
    soup = url_to_soup(mk_param_url(category_page_url, page))

    # A button with load more appears, when
    while soup.find(class_="search__loadmore js-loadmorecontainer") is not None:
        logger.debug(f"Extracting more pages for: {category_page_url}")
        page += 1
        soup = url_to_soup(mk_param_url(category_page_url, page))

    product_links = [item["href"] for item in soup.find_all(class_="thumb-link")]

    logger.debug(f"Found {len(product_links)} products in: {category_page_url}")

    return product_links


MAIN_IMG_BLOCK = "pdp-main__blocks-box__item__inner"
PROD_IMAGE_HTML_CLASS = "pdp-main__blocks-box__item__visual"
RATING_ELEMENT = "bv_avgRating_component_container"
RATING_COUNT_ELEMENT = "bv_numReviews_text"
TITLE_ELEMENT = "product-name"
DESCRIPTION_ELEMENT = (
    "pdp-main__blocks-box__item pdp-main__blocks-box__item--description xx"
)


def try_extract_product_info(
    product_url_info: ProductURLInfo,
    soup: BeautifulSoup,
    allow_missing_description: bool = False,
) -> ProductInfo:
    """
    Try to extract all product information given `soup` of product URL.
    A `MissingProductInfoException` is risen when expected information could not be found.
    """
    main_img_block = soup.find(class_=MAIN_IMG_BLOCK)

    if main_img_block is None:
        raise MissingProductInfoException("Main image block is missing")

    # main_image_url = main_img_block.find(name="img")["src"]
    main_image_url = main_img_block.find(name="img")["src"]

    if main_image_url is None:
        raise MissingProductInfoException("Main image URL not found")

    # Find all product image urls
    all_image_urls = [
        e["data-src"] for e in soup.find_all(name="img", class_=PROD_IMAGE_HTML_CLASS)
    ]

    # Filter those that are not the main image, and make sure the urls are unique
    other_image_urls = list(set(url for url in all_image_urls if url != main_image_url))

    # Extract rating count
    rating_count_elem = soup.find(class_=RATING_COUNT_ELEMENT)
    if rating_count_elem is None:
        raise MissingProductInfoException("Missing rating count element")

    rating_count_text: str = rating_count_elem.get_text()

    if rating_count_text == "":
        raise MissingProductInfoException("Rating count text not yet loaded")

    rating_count = int(rating_count_text.strip("()"))

    # Extract average rating if there is a rating
    if rating_count > 0:
        rating_elem = soup.find(class_=RATING_ELEMENT)
        if rating_elem is None:
            raise MissingProductInfoException("Missing rating element")

        rating_text: str = rating_elem.get_text()

        if rating_text == "":
            raise MissingProductInfoException("Rating text not yet loaded")

        rating = float(rating_text)
    else:
        rating = None

    # Extract title
    title_element = soup.find(class_=TITLE_ELEMENT)
    if title_element is None:
        raise MissingProductInfoException("Missing title element")

    title = title_element.get_text()

    if ONLY_WHITE_SPACE.match(title):
        raise MissingProductInfoException("Empty title text")

    # Extract description
    description_elem = soup.find(class_=DESCRIPTION_ELEMENT)

    # Description is sometimes not present, it is only skipped if it is not present on last try
    if description_elem is None:
        if allow_missing_description:
            description = None
        else:
            raise MissingProductInfoException("Missing description element")
    else:
        description = description_elem.get_text()

        if bool(ONLY_WHITE_SPACE.match(description)) and not allow_missing_description:
            raise MissingProductInfoException("Missing description text")

    return ProductInfo(
        url=product_url_info.product_url,
        main_image_url=main_image_url,
        other_image_urls=other_image_urls,
        category=product_url_info.category,
        gender=product_url_info.gender,
        rating=rating,
        rating_count=rating_count,
        title=title,
        description=description,
    )


def extract_product_info(
    product_url_info: ProductURLInfo,
    driver: Optional[WebDriver] = None,
    max_tries: int = 3,
    load_time_min: float = 1.0,
) -> ProductInfo:
    """Extract product information from Intersport product page

    ## Parameters:
    - `driver` (`WebDriver`): The selenium webdriver used to load the page
    - `product_url` (`str`): The url to the Intersport product page
    - `max_tries` (`int`): Maximum amount of tries to extract product information. Each time, the page is given twice the amount of time to load.
    - `load_time_min` (`float`): Amount of time given for page to load in seconds on first try. Each subsequent try, this time is doubled

    ## Throws:
    - `NoSuchElementException`: When a webpage element could not be found

    ## Returns
    - `ProductInfo`: Information of a product

    """
    load_time = load_time_min
    tries = 1

    failure_cause = None

    while True:
        if tries > max_tries:
            raise MissingProductInfoException(
                f"Could not collect product information after max tries, {failure_cause=}"
            )

        # Load the page
        soup = url_to_soup(
            product_url_info.product_url, load_time_sec=load_time, driver=driver
        )

        try:
            result = try_extract_product_info(
                product_url_info, soup, allow_missing_description=tries == max_tries
            )
            logger.info(
                f"Successfully extracted info for: {product_url_info.product_url}"
            )
            return result
        except MissingProductInfoException as e:
            logger.warning(
                f"Missing product info ({e}), {load_time=}, {product_url_info.product_url=}"
            )
            failure_cause = e
            load_time *= 2
            tries += 1


def extract_intersport_product_infos(
    product_urls: List[ProductURLInfo],
) -> List[ProductInfo]:
    """Extract product information for the product URLs

    ## Parameters:
    - `product_urls` (`List[ProductURLInfo]`): The list of intersport product URLs and meta information


    ## Returns
    - `List[ProductInfo]`: Extracted product information

    """
    # Create driver to use for this chunk
    driver = get_headless_firefox_driver()

    result = []
    for url_info in product_urls:
        try:
            info = extract_product_info(url_info, driver)
            result.append(info)
        except Exception as e:
            logger.warning(f"Failed extracting product: {e}: {url_info}")

    driver.quit()
    return result


@click.command()
@click.option(
    "--out-file",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="File to store extracted data in parquet format",
)
@click.option(
    "--url-out-file",
    required=False,
    type=click.Path(path_type=Path),
    help="Optional text file to store urls",
)
@click.option(
    "--num-jobs",
    "-j",
    type=click.INT,
    default=multiprocessing.cpu_count(),
    help="Amount of jobs (requests) to do simultaneaously, mostly IO bound",
)
def main(out_file: Path, url_out_file: Optional[Path], num_jobs: int):
    """Extracts shoe product information from the intersport website and stores this in a parquet file.

    To see which information is extracted, see the `ProductInfo` class.

    ## Parameters:
    - `out_file` (`Path`): Output parquet file path
    - `url_out_file` (`Optional[Path]`): Optional text file in which to store product urls
    - `num_jobs` (`int`): Amount of simultateous extractions performed.

    """

    # Extract all products of all gender/category combinations in parallel
    logger.info("Extracting product URLs")

    def extract_product_url_task(gen, cat) -> List[ProductURLInfo]:
        return [
            ProductURLInfo(url, cat, gen)
            for url in extract_intersport_product_urls(gen, cat)
        ]

    all_products_stacked = joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(extract_product_url_task)(gender, cat)
        for gender, cat in itertools.product(GENDERS, SHOE_CATEGORIES)
    )

    # List of all products
    all_product_urls = list(flatten(all_products_stacked))

    # Store the urls
    if url_out_file is not None:
        url_out_file.parent.mkdir(exist_ok=True, parents=True)

        with open(str(url_out_file), "w") as f:
            f.write(
                os.linesep.join(
                    orjson.dumps(attrs.asdict(u)).decode() for u in all_product_urls
                )
                + os.linesep
            )

    logger.info(f"Found {len(all_product_urls)} products")

    # Split products into `num_jobs` batches, necessary to properly share a webdriver
    all_products_chunked = list(split_into_n_chunks(all_product_urls, num_jobs))

    # Extract product information in parallel
    results_stacked = joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(extract_intersport_product_infos)(urls)
        for urls in all_products_chunked
    )

    results = list(flatten(results_stacked))

    logger.info(
        f"Successfully extracted information of {len(results)} / {len(all_product_urls)} products"
    )

    # Create dataframe
    result_df = pl.DataFrame([attrs.asdict(r) for r in results])

    # Store results
    out_file.parent.mkdir(exist_ok=True, parents=True)
    result_df.write_parquet(out_file)


if __name__ == "__main__":
    main()
