import re
import ast
import json
from pydantic import BaseModel, Field
from typing import Literal, Optional

class AirQuerySchema(BaseModel):
    query_type: Literal["current", "forecast"] = Field(
        description="Pick one of the air_quality query_type: 'current' or 'forecast'"
    )
    city_name: str = Field(description="City Name")
    country_code: str = Field(description="Country Code in ISO-3166-1 alpha-2")
    days: Optional[int] = Field(
        7, description="This field is required when using forecast"
    )

def air_query_prompt(user_input: str) -> str:
    return (
        "Extract structured information from this air quality query.\n"
        "Output ONLY a JSON object with: query_type ('current' or 'forecast'), city_name, country_code (ISO-2), days (integer, required only for 'forecast').\n"
        "Several examples:\n"
        "User query: 'What is air quality in Delhi, India today?'\n"
        "Output: {\"query_type\": \"current\", \"city_name\": \"Delhi\", \"country_code\": \"IN\"}\n"
        "User query: 'Air quality of Pune, India for next 3 days'\n"
        "Output: {\"query_type\": \"forecast\", \"city_name\": \"Pune\", \"country_code\": \"IN\", \"days\": 3}\n"
        "User query: 'What will be the air quality in San Francisco for the weekend?'\n"
        "Output: {\"query_type\": \"forecast\", \"city_name\": \"San Francisco\", \"country_code\": \"US\", \"days\": 2}\n"
        "User query: " + user_input + "\n"
        "Output:"
    )

def get_air_query_inputs(user_prompt: str, model) -> AirQuerySchema:
    prompt = air_query_prompt(user_prompt)
    result = model.invoke(prompt)
    text = result.content.strip()

    # Extract JSON or Python dict string
    try:
        if text.startswith("{") and text.endswith("}"):
            data = json.loads(text.replace("'", '"'))  # replace single quotes if needed
        else:
            # Try extracting from markdown block if present
            match = re.search(r"\{.*\}", text)
            data = json.loads(match.group().replace("'", '"')) if match else ast.literal_eval(text)
        return AirQuerySchema(**data)
    except Exception as e:
        print("Failed to parse schema:", e)
        return None