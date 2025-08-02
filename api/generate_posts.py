import json
import random
import os
from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict
import uuid
from datetime import datetime
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
CONFIG = {
    'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY'),
    'post_count_per_day': 4,
    'content_topics': [
        'Emerging Technologies', 'Industry Trends', 'Product Innovation',
        'Software Engineering', 'AI Developments'
    ],
    'post_templates': [
        "Sharing insights on {topic}: {content} This advancement is shaping the future of technology. #Tech #Innovation",
        "The latest in {topic}: {content} This development highlights the potential for growth in the industry. #Technology #Future",
        "A deep dive into {topic}: {content} This trend is driving significant progress. #TechTrends #Innovation",
        "Exploring advancements in {topic}: {content} These innovations are redefining the landscape. #SoftwareEngineering #Tech",
        "{topic} continues to evolve: {content} This progress underscores the impact of technology on our world. #AI #Innovation"
    ],
    'output_dir': '/tmp/pending_posts',
    'manual_review': True  # Set to False for Zapier/IFTTT auto-posting
}

# Ensure output directory exists
def ensure_output_dir():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

def generate_content_with_huggingface(topic: str, model: str = "google/flan-t5-large") -> str:
    """Generate content using Hugging Face Inference API with fallback."""
    if not CONFIG['huggingface_api_key']:
        logger.error("HUGGINGFACE_API_KEY is not set.")
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY is not configured.")
    
    try:
        headers = {
            'Authorization': f'Bearer {CONFIG["huggingface_api_key"]}',
            'Content-Type': 'application/json'
        }
        prompt = (
            f"Write a detailed, professional LinkedIn post (300-500 words) about {topic}. "
            "Use a positive, informative tone, focus on one key insight or advancement, and align with a tech-savvy "
            "personal brand. Avoid jargon, provide specific details, examples, or use cases, and do not include questions "
            "or calls-to-action. Include relevant statistics, trends, or real-world applications to enhance depth."
        )
        payload = {
            'inputs': prompt,
            'parameters': {'min_length': 300, 'max_length': 512, 'temperature': 0.7}
        }
        response = requests.post(
            f'https://api-inference.huggingface.co/models/{model}',
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        generated_text = response.json()[0]['generated_text'].strip()
        
        # If content is too short, try a follow-up generation
        if len(generated_text.split()) < 300:
            follow_up_prompt = (
                f"Continue the previous LinkedIn post about {topic}, adding 200-300 more words. "
                "Maintain a positive, informative tone, provide additional details, examples, or applications, "
                "and align with a tech-savvy personal brand. Avoid jargon and do not include questions or calls-to-action."
            )
            payload['inputs'] = follow_up_prompt
            follow_up_response = requests.post(
                f'https://api-inference.huggingface.co/models/{model}',
                json=payload,
                headers=headers
            )
            follow_up_response.raise_for_status()
            follow_up_text = follow_up_response.json()[0]['generated_text'].strip()
            generated_text = f"{generated_text} {follow_up_text}"
        
        # Truncate to ~3000 characters to fit LinkedIn's limit
        return generated_text[:3000]
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error from Hugging Face API ({model}): {e.response.status_code} - {e.response.text}")
        if model != "t5-small":  # Try fallback model
            logger.info("Switching to fallback model t5-small")
            return generate_content_with_huggingface(topic, model="t5-small")
        return (
            f"Exploring {topic} today! Recent advancements are driving significant progress in this field. "
            "Innovations are enabling new applications, from streamlining processes to enhancing scalability. "
            "For example, recent developments have led to a 20% improvement in processing efficiency in related systems. "
            "This trend is poised to influence industries by improving operational workflows and fostering sustainable growth."
        )
    except Exception as e:
        logger.error(f"Error generating content with Hugging Face ({model}): {str(e)}")
        if model != "t5-small":  # Try fallback model
            logger.info("Switching to fallback model t5-small")
            return generate_content_with_huggingface(topic, model="t5-small")
        return (
            f"Exploring {topic} today! Recent advancements are driving significant progress in this field. "
            "Innovations are enabling new applications, from streamlining processes to enhancing scalability. "
            "For example, recent developments have led to a 20% improvement in processing efficiency in related systems. "
            "This trend is poised to influence industries by improving operational workflows and fostering sustainable growth."
        )

def create_post() -> Dict:
    """Create a single LinkedIn post."""
    topic = random.choice(CONFIG['content_topics'])
    content = generate_content_with_huggingface(topic)
    template = random.choice(CONFIG['post_templates'])
    post_text = template.format(topic=topic, content=content)
    return {
        'id': str(uuid.uuid4()),
        'text': post_text,
        'topic': topic,
        'created_at': datetime.now().isoformat()
    }

def save_posts_for_review(posts: List[Dict]):
    """Save generated posts to a JSON file for manual review."""
    file_path = Path(CONFIG['output_dir']) / f'posts_{datetime.now().strftime("%Y%m%d")}.json'
    try:
        with open(file_path, 'w') as f:
            json.dump(posts, f, indent=2)
        logger.info(f"Saved {len(posts)} posts to {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving posts: {e}")
        return None

def post_to_zapier(post: Dict):
    """Send post to Zapier webhook for LinkedIn publishing (optional)."""
    try:
        zapier_webhook = os.getenv('ZAPIER_WEBHOOK')
        if not zapier_webhook:
            logger.error("ZAPIER_WEBHOOK is not set.")
            return
        payload = {'text': post['text']}
        response = requests.post(zapier_webhook, json=payload)
        response.raise_for_status()
        logger.info(f"Sent post {post['id']} to Zapier webhook.")
    except Exception as e:
        logger.error(f"Error sending to Zapier: {e}")

def load_pending_posts() -> List[Dict]:
    """Load posts from the latest JSON file for review or posting."""
    today = datetime.now().strftime("%Y%m%d")
    file_path = Path(CONFIG['output_dir']) / f'posts_{today}.json'
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        logger.info("No pending posts found.")
        return []

def post_to_linkedin(post_text: str, access_token: str, author_urn: str) -> Dict:
    """Post content directly to LinkedIn using the Posts API."""
    try:
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0',
            'LinkedIn-Version': '202508'  # Use current version (YYYYMM)
        }
        payload = {
            'author': author_urn,
            'commentary': post_text,
            'visibility': 'PUBLIC',
            'distribution': {
                'feedDistribution': 'MAIN_FEED',
                'targetEntities': [],
                'thirdPartyDistributionChannels': []
            },
            'lifecycleState': 'PUBLISHED',
            'isReshareDisabledByAuthor': False
        }
        response = requests.post(
            'https://api.linkedin.com/rest/posts',
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        post_id = response.headers.get('x-restli-id', 'Unknown')
        logger.info(f"Successfully posted to LinkedIn. Post ID: {post_id}")
        return {
            'status': 'success',
            'post_id': post_id,
            'message': 'Post uploaded to LinkedIn'
        }
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error from LinkedIn API: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"LinkedIn API error: {e.response.text}")
    except Exception as e:
        logger.error(f"Error posting to LinkedIn: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to post to LinkedIn: {str(e)}")

@app.post("/generate-posts")
async def generate_posts():
    """Generate and save posts for the day."""
    ensure_output_dir()
    posts = [create_post() for _ in range(CONFIG['post_count_per_day'])]
    
    if CONFIG['manual_review']:
        file_path = save_posts_for_review(posts)
        return {
            "message": "Posts generated and saved for review",
            "posts": posts,
            "file_path": file_path or "Error saving file"
        }
    else:
        for post in posts:
            post_to_zapier(post)
        return {
            "message": "Posts generated and sent to Zapier",
            "posts": posts
        }

@app.get("/get-pending-posts")
async def get_pending_posts():
    """Retrieve pending posts for manual review."""
    posts = load_pending_posts()
    return {
        "message": "Retrieved pending posts",
        "posts": posts
    }

@app.post("/upload-post")
async def upload_post(
    post_id: str = Body(...),
    access_token: str = Body(...),
    author_urn: str = Body(...)
):
    """Manually upload a pending post to LinkedIn by post ID."""
    posts = load_pending_posts()
    post = next((p for p in posts if p['id'] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post ID not found")
    
    result = post_to_linkedin(post['text'], access_token, author_urn)
    return {
        "message": result['message'],
        "post_id": result['post_id'],
        "original_post": post
    }