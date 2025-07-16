# Web caption generation prompt
# Instructs the model to generate meta description content for web pages
WEB_CAPTION_PROMPT = """You are given a screenshot of a webpage. Please generate the meta web description information of this webpage, i.e., content attribute in <meta name="description" content=""> HTML element.

You should use the following format, and do not output any explanation or any other contents:
<meta name="description" content="YOUR ANSWER">
"""

# Heading OCR prompt
# Instructs the model to extract main heading text from web page screenshots
HEADING_OCR_PROMPT = """You are given a screenshot of a webpage. Please generate the main text within the screenshot, which can be regarded as the heading of the webpage.

You should directly tell me the main content, and do not output any explanation or any other contents.
"""

# Web question answering prompt
# Template for asking questions about web page content
WEBQA_PROMPT = """{question}
You should directly tell me your answer in the fewest words possible, and do not output any explanation or any other contents.
"""

# Element OCR prompt
# Instructs the model to perform OCR on specific bounding box regions
ELEMENT_OCR_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please perform OCR in the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""

# Element grounding prompt
# Instructs the model to select the correct element from labeled candidates
ELEMENT_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one best matches the description: {element_desc}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

# Action prediction prompt
# Instructs the model to predict what happens when clicking on an element
ACTION_PREDICTION_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:
{choices_text}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

# Action grounding prompt
# Instructs the model to select the correct element to complete a task
ACTION_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one I should click to complete the following task: {instruction}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

# Default prompts dictionary
# Maps task types to their corresponding default prompts
DEFAULT_PROMPTS = {
    "web_caption_prompt": WEB_CAPTION_PROMPT,
    "heading_ocr_prompt": HEADING_OCR_PROMPT,
    "webqa_prompt": WEBQA_PROMPT,
    "element_ocr_prompt": ELEMENT_OCR_PROMPT,
    "element_ground_prompt": ELEMENT_GROUND_PROMPT,
    "action_prediction_prompt": ACTION_PREDICTION_PROMPT,
    "action_ground_prompt": ACTION_GROUND_PROMPT,
}
