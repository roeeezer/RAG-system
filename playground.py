import sys
import codecs
import os

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)


def test_csv_file_format():
    current_directory_path = os.getcwd()
    tests_outputs_folder_name = "part1 files"
    file_name = "queries.csv"
    file_path = os.path.join(current_directory_path, tests_outputs_folder_name, file_name)
    lines = open(file_path, encoding='utf-8').readlines()
    for line in lines:
        print(line)

def get_api_key():
    current_directory_path = os.getcwd()
    tests_outputs_folder_name = "untracked"
    file_path = os.path.join(current_directory_path, tests_outputs_folder_name, "gemini_api_key.txt")
    api_key = open(file_path, encoding='utf-8').read()
    return api_key

def test_gemini_api_call():
    import google.generativeai as genai
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("What is the best fruit?")
    print(response.text)

def main():
    test_gemini_api_call()

if __name__ == "__main__":
    main()

