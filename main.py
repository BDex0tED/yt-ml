import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from pytube import YouTube

import time
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, TooManyRequests



load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("Error googleapikey wasn't found in environment variables.")
    exit()

VIDEO_ID = "8bMh8azh3CY"

yt = YouTube("https://www.youtube.com/watch?v=Zi_XLOBDo_Y")

caption = yt.captions.get_by_language_code("en")
text = caption.generate_srt_captions()

print(text)

EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
CHAT_MODEL = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)


def get_video_text(video_id):
    """Downloads subtitles for a YouTube video"""
    print(f"Downloading subtitles for video {video_id}...")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["ru", "en"])
        full_text = " ".join([line["text"] for line in transcript])
        return full_text
    except Exception as e:
        print(f"Error downloading subtitles: {e}")
        return None


def create_rag_system(text):
    """Creates a knowledge base from text"""
    print("🔪 Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"There's {len(chunks)} pieces.")

    print("💾 Vectorizing and saving to ChromaDB...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=EMBEDDING_MODEL,
        collection_name="youtube_data"
    )

    retriever = vectorstore.as_retriever()

    def format_prompt(inputs):
        return f"""
Ответь строго по контексту.

Контекст:
{inputs['context']}

Вопрос:
{inputs['question']}
"""

    qa_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | format_prompt
        | CHAT_MODEL
        | StrOutputParser()
    )

    return qa_chain

def get_video_text(video_id):
    print(f"Downloading subtitles for video {video_id}...")

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            transcript = transcripts.find_transcript(['en'])
        except:
            transcript = transcripts.find_generated_transcript(['en'])

        # 🔥 КРИТИЧЕСКОЕ МЕСТО
        # YouTube иногда ломает manual EN, но translate всегда работает
        try:
            data = transcript.fetch()
        except:
            data = transcript.translate('en').fetch()

        full_text = " ".join([line["text"] for line in data])
        return full_text

    except Exception as e:
        print("Error downloading subtitles:", e)
        return None
    

def safe_get_transcript(video_id):
    for attempt in range(5):
        try:
            # пробуем сначала английские субтитры
            return YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=["en", "en-US", "en-GB"]
            )
        except TooManyRequests:
            print(f"Rate limit! Attempt {attempt+1}/5... sleep 3s")
            time.sleep(3)
        except (TranscriptsDisabled, NoTranscriptFound):
            print("No subs on this video.")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    print("Failed after retries.")
    return None


video_id = "Zi_XLOBDo_Y"  # подставь свой
transcript = safe_get_transcript(video_id)

if transcript:
    text = " ".join([x["text"] for x in transcript])
    print("\nSUCCESS\n")
    print(text[:500], "...")




if __name__ == "__main__":
    text = get_video_text(VIDEO_ID)

    if text:
        qa_chain = create_rag_system(text)

        print("\nSystem is ready to chat!")
        print("(Type 'exit' to quit)\n")

        while True:
            query = input("Your question: ")
            if query.lower() in ["exit", "quit"]:
                break

            response = qa_chain.invoke(query)
            print(f"\n🤖 Gemini: {response}\n")
            print("-" * 50)
