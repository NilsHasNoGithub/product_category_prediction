from typing import Optional

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.remote.webdriver import WebDriver

from .definitions import OptKwargs

_BS_HTML_PARSER = "html.parser"


class UnexpectedStatusCodeException(Exception):
    """Exception raised when status code of request is unexpected"""


class MissingProductInfoException(Exception):
    """Exception raised when product info is missing on a webpage"""


def url_to_soup(
    url: str,
    load_time_sec: float = 1.0,
    driver: Optional[WebDriver] = None,
    bs_kwargs: OptKwargs = None,
) -> BeautifulSoup:
    """
    Create Beautiful Soup from URL.
    `driver`: Optional selenium driver to use. If not provided, a new, headless firefox instance is used. A single `get` requests using the `requests` library is not enough to obtain all relevant elements in the page.
    `bs_kwargs`: Keyword arguments passed to `Beautifulsoup`
    """
    if bs_kwargs is None:
        bs_kwargs = dict()

    no_driver = driver is None

    # If no driver is provided, create a driver
    if no_driver:
        driver = get_headless_firefox_driver()

    try:
        driver.get(url)
        # Wait for page to properly load
        driver.implicitly_wait(load_time_sec)
        page_content = driver.page_source
    finally:
        # Quit driver if it was not provided
        if no_driver:
            driver.quit()

    # Create the soup
    soup = BeautifulSoup(page_content, _BS_HTML_PARSER, **bs_kwargs)

    return soup


def get_headless_firefox_driver() -> WebDriver:
    """Create a headless Selenium firefox instance"""
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    return driver
