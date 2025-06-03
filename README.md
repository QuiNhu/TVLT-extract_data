# Legal Articles Extractor API

This API provides two main endpoints to extract legal articles from `.txt` files:

1. **/v1/articles\_extractor** — Extracts all legal articles and saves the output as a JSON file.
2. **/v1/relevant\_articles\_extractor** — Extracts legal articles relevant to a user’s question.

---

## Requirements

* Python 3.10+
* FastAPI and required libraries
* Implemented functions: `legal_articles_extractor(file_path: str)` and `get_relevant_articles(question: str, articles: dict)`

---

## Usage

### 1. Extract All Legal Articles

**POST** `/v1/articles_extractor`

**Request Body (JSON):**

```json
{
  "txt_path": "path/to/file.txt"
}
```

* `txt_path`: Valid path to the input `.txt` file.

**Success Response:**

```json
{
  "message": "Extraction successful",
  "output_file": "path/to/file.json"
}
```

**Common Errors:** Missing `txt_path`, file not found, invalid file extension, or failure saving JSON.

---

### 2. Extract Relevant Articles Based on a Question

**POST** `/v1/relevant_articles_extractor`

**Request Body (JSON):**

```json
{
  "txt_path": "path/to/file.txt",
  "question": "Your legal question here"
}
```

**Success Response:**

```json
{
  "message": "Extract relevant articles success!!",
  "file_path": "file.txt",
  "queries": {
    "title": "Relevant article title",
    "text": "Relevant article content"
  }
}
```

**Common Errors:** Missing or invalid `txt_path`, or missing `question`.

---

## Example using curl

Extract all articles:

```bash
curl -X POST http://localhost:8000/v1/articles_extractor \
-H "Content-Type: application/json" \
-d '{"txt_path": "/path/to/file.txt"}'
```

Extract relevant articles:

```bash
curl -X POST http://localhost:8000/v1/relevant_articles_extractor \
-H "Content-Type: application/json" \
-d '{"txt_path": "/path/to/file.txt", "question": "Labor contract regulations"}'
```

---

## Notes

* Ensure the `.txt` file exists and is readable by the server.
* Output JSON file is saved alongside the input file with the same name but `.json` extension.
* The extraction functions must be properly implemented.

---

For support, please contact the development team.
