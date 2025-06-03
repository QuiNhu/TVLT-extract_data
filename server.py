import boto3
import os
import json

from typing import Text
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from relevant_extractor.domain.logging_utils import get_logger
from relevant_extractor.application import legal_articles_extractor, get_relevant_articles


s3 = boto3.client("s3")

app = FastAPI(
    title="TVPL Test Server",
    version="1.0",
    description="Extract relevant law articles in Vietnamese",
    debug=True,
)

s3_client = boto3.client('s3')

# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

logger = get_logger(__name__)


# noinspection PyUnusedLocal
@app.post("/v1/articles_extractor", include_in_schema=False)
async def articles_extractor(request: Request) -> JSONResponse:
    """
    Endpoint to extract legal articles from a given .txt file.

    Parameters:
      request (Dict): {
        "txt_path": Path to the input .txt file containing legal text.
      }

    Returns:
      JSONResponse with extraction status and output JSON file path, or error message with appropriate HTTP status code.
    """

    data = await request.json()
    txt_path = data.get("txt_path")
    # Validate presence of the txt_path parameter
    if txt_path is None:
        return JSONResponse(
            content={"error": "Missing file path (.txt)"},
            status_code=400
        )

    # Validate file extension
    if not txt_path.endswith(".txt"):
        return JSONResponse(
            content={"error": "Invalid file path (.txt) extension required"},
            status_code=400
        )

    # Check if the specified file exists
    if not os.path.exists(txt_path):
        return JSONResponse(
            content={"error": "File path not found (.txt)"},
            status_code=404
        )

    # Extract legal articles from the input file
    articles_dict = await legal_articles_extractor(file_path=txt_path)

    # Define output JSON file path by replacing .txt extension with .json
    output_json_path = f"{txt_path[:-4]}.json"

    # Attempt to save the extracted articles to a JSON file
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(
                articles_dict,
                f,
                ensure_ascii=False,
                indent=4
            )
    except Exception as e:
        # Return HTTP 500 if file saving fails
        return JSONResponse(
            content={"error": f"Failed to save JSON file: {str(e)}"},
            status_code=500
        )

    # Return success response with the path to the saved JSON file
    return JSONResponse(
        content={
            "message": "Extraction successful",
            "output_file": output_json_path
        }
    )


# noinspection PyUnusedLocal
@app.post("/v1/relevant_articles_extractor", include_in_schema=False)
async def relevant_articles_extractor(request: Request) -> JSONResponse:
    data = await request.json()
    question = data.get("question")
    file_path = data.get("txt_path")
    if not file_path:
        return JSONResponse(
            content={
                "message": "Empty file!!",
                "file_path": None,
                "queries": {
                    "title": None,
                    "text": None
                }
            }
        )

    file_name = file_path.split("/")[-1]
    if not file_path.lower().endswith(".txt"):
        return JSONResponse(
            content={
                "message": "Invalid file path!!",
                "file_path": file_name,
                "queri es": {
                    "title": None,
                    "text": None
                }
            }
        )

    if not question:
        return JSONResponse(
            content={
                "message": "Question Empty! Can not extract relevant law articles!!",
                "file_path": file_name,
                "queries": {
                    "title": None,
                    "text": None
                }
            }
        )

    articles_dict = await legal_articles_extractor(file_path=file_path)
    response = await get_relevant_articles(question=question, articles=articles_dict)
    return JSONResponse(
        content={
            "message": "Extract relevant articles success!!",
            "file_path": file_name,
            "queries": response
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50)