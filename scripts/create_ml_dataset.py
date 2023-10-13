"""Script to turn extracted product data into a local dataset, such that it does not depend on external resources"""
import copy
import multiprocessing
from pathlib import Path
from typing import Any, Dict

import click
import joblib
import orjson
import polars as pl
import requests
from loguru import logger
from tqdm import tqdm

from product_prediction.scraping import UnexpectedStatusCodeException
from product_prediction.utils import base64_encode


def download_img_to_b64(image_url: str) -> str:
    """Downloads image and encodes it as base 64

    ## Parameters:
    - `image_url` (`str`): url to image

    ## Returns
    - `str`: base64 encoded PNG
    """

    resp = requests.get(image_url)

    if resp.status_code != 200:
        raise UnexpectedStatusCodeException(
            "Did not get status code 200 when downloading image"
        )

    return base64_encode(resp.content)


def process_sample(dataset_out_dir: Path, index: int, data_row: Dict[str, Any]) -> None:
    """Downloads images from data row and stores all data in a json under the `category` directory in `out_dir`

    ## Parameters:
    - `dataset_out_dir` (`Path`): Dir where dataset is stored
    - `index` (`int`): index of this sample
    - `data_row` (`Dict[str, Any]`): row from dataframe containing sample (meta)data
    """

    # path to store sample
    out_file: Path = dataset_out_dir / data_row["category"] / f"{index:06d}.json"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    try:
        main_img_b64 = download_img_to_b64(data_row["main_image_url"])
    except Exception as e:
        logger.warning(f"Failed downloading main image ({e}), skipping sample: {url}")

    other_img_b64s = []

    for url in data_row["other_image_urls"]:
        try:
            img = download_img_to_b64(url)
            other_img_b64s.append(img)
        except Exception as e:
            logger.warning(
                f"Failed downloading one of alternative product images ({e}), for {url}"
            )

    json_data = copy.copy(data_row)

    json_data["main_image_base64"] = main_img_b64
    json_data["other_images_base64"] = other_img_b64s

    with open(str(out_file), "wb") as f:
        f.write(orjson.dumps(json_data))


@click.command()
@click.option(
    "--in-file",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to extracted product data (parquet file, output of `extract_plutosport_products.py)",
)
@click.option(
    "--out-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory to store results",
)
@click.option(
    "--num-jobs",
    "-j",
    default=multiprocessing.cpu_count(),
    help="Amount of simultaneous jobs",
)
def main(in_file: Path, out_dir: Path, num_jobs: int):
    out_dir.mkdir(exist_ok=True, parents=True)

    data = pl.read_parquet(in_file)

    # Each sample is stored as a separate json file, with images encoded using base64
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(process_sample)(out_dir, idx, row)
        for idx, row in enumerate(tqdm(data.iter_rows(named=True), total=len(data)))
    )


if __name__ == "__main__":
    main()
