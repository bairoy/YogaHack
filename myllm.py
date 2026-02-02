from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ Define LLM
llm = ChatOpenAI(model_name="gpt-5-nano-2025-08-07")

# 2️⃣ Strict, professional coaching prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional yoga coach. "
     "You are given: "
     "- Pose name, "
     "- Misaligned joints, "
     "- Pose accuracy percentage. "
     "Your output must be: "
     "- Short, precise, actionable instructions for correction. "
     "- Professional and encouraging tone. "
     "- Do NOT add extra explanations, background, or unrelated advice. "
     "- Each instruction should be concise, clear, and easy to follow."),
     
    ("user", "{input}")  # e.g., "Pose: Warrior II, Misaligned joints: left_knee, right_elbow, Accuracy: 76%"
])

# 3️⃣ Chain with parser
chain = prompt | llm | StrOutputParser()
