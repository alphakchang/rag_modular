from typing import List
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os

def load_html_basic(file_path: str) -> List:

    print(f"Loading {file_path}...", end="")
    loader = UnstructuredHTMLLoader(file_path)
    docs = loader.load()
    print("Done")
    return docs

def load_html_from_urls(links: List[str]) -> List:

    print(f"Loading data from links..", end="")
    
    loader = UnstructuredURLLoader(urls=links)
    data = loader.load()

    print("Done")
    
    return data

def load_html_moodle(links: List[str]) -> List:
    driver = webdriver.Chrome()
    driver.get("https://moodle.alphacrc.com:5081/")
    assert "Alpha" in driver.title

    login_elem = driver.find_element(By.NAME, "username")
    password_elem = driver.find_element(By.NAME, "password")
    login_btn_elem = driver.find_element(By.ID, "loginbtn")

    login_elem.send_keys(os.getenv('MOODLE_USERNAME'))
    password_elem.send_keys(os.getenv('MOODLE_PASSWORD'))

    login_btn_elem.send_keys(Keys.RETURN)

    page_contents = []
    for link in links:
        driver.get(link)

        content_elem = driver.find_element(By.ID, "mod_book-chapter")
        
        page_contents.append(content_elem.text)

    final_contents = " ".join(page_contents)

    driver.close()

    return final_contents