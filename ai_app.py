from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import List, Optional
from datetime import datetime, timedelta

#set up 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

#FastAPI app
app = FastAPI(title="NITDA Staff Assistant AI", version="1.0")

# models

class PromptRequest(BaseModel):
    prompt: str

class SingleEmailRequest(BaseModel):
    email_text: str

class MultipleEmailRequest(BaseModel):
    emails: List[str]

class ReportRequest(BaseModel):
    summary: Optional[str] = None
    report_points: Optional[List[str]] = None
    author: Optional[str] = "NITDA Staff"
    date: Optional[str] = None

class MemoRequest(BaseModel):
    context: str

class ReminderRequest(BaseModel):
    task: str
    date_time: str   # format: "YYYY-MM-DD HH:MM"
    priority: Optional[str] = "Normal"

class EventRequest(BaseModel):
    title: str
    date_time: str   # format: "YYYY-MM-DD HH:MM"
    duration_minutes: Optional[int] = 60
    description: Optional[str] = None
    attendees: Optional[List[str]] = []

#DATABASE (in-memory)

reminders_db = []
calendar_db = []

#ROUTES

@app.get("/", tags=["AI"])
def home():
    """Simple health check"""
    return {"message": "🚀 AI Assistant is running!"}

#General AI 
@app.post("/ask", tags=["AI"])
def ask_ai(request: PromptRequest):
    """General Q&A with the assistant"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful staff assistant."},
                {"role": "user", "content": request.prompt}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Emails
@app.post("/sort_email", tags=["Emails"])
async def sort_email(request: SingleEmailRequest):
    """Sort a single email into High / Medium / Low with explanation"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that sorts and prioritizes staff emails."},
                {"role": "user", "content": f"Analyze this email and classify its priority (High, Medium, Low). "
                                            f"Explain briefly why.\n\nEmail:\n{request.email_text}"}
            ],
        )
        return {"priority_result": response.choices[0].message["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sort_emails", tags=["Emails"])
async def sort_emails(request: MultipleEmailRequest):
    """Sort multiple emails by urgency/importance with summary"""
    try:
        prompt = (
            "You are an AI assistant that helps staff prioritize emails. "
            "Sort the following emails by urgency and importance, and return a summary with priority labels "
            "(High, Medium, Low).\n\n"
            f"Emails:\n{request.emails}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return {"sorted_emails": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Reports
@app.post("/generate_report", tags=["Reports"])
async def generate_report(request: ReportRequest):
    """Generate a professional staff report"""
    try:
        report_date = request.date or datetime.today().strftime("%B %d, %Y")

        if request.summary:
            user_content = f"Generate a professional staff report based on this summary:\n{request.summary}"
        elif request.report_points:
            user_content = f"Generate a professional staff report based on these points:\n{request.report_points}"
        else:
            raise HTTPException(status_code=400, detail="Provide either 'summary' or 'report_points'.")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that generates professional and official staff reports. "
                        "Always structure the report with a title, date, author, introduction, sections, and conclusion."
                    )
                },
                {"role": "user", "content": f"Author: {request.author}\nDate: {report_date}\n\n{user_content}"}
            ],
        )
        return {"report": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#memos
@app.post("/draft_memo", tags=["Memos"])
async def draft_memo(request: MemoRequest):
    """Draft a professional staff memo"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that drafts professional and official staff memos. "
                        "Always use a formal structure with To, From, Date, and Subject lines."
                    )
                },
                {"role": "user", "content": f"Draft a memo for this:\n{request.context}"}
            ],
        )
        return {"memo_draft": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Reminders
@app.post("/create_reminder", tags=["Reminders"])
async def create_reminder(request: ReminderRequest):
    """Create and store a reminder"""
    try:
        try:
            reminder_time = datetime.strptime(request.date_time, "%Y-%m-%d %H:%M")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD HH:MM")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that creates helpful, professional staff reminders."},
                {"role": "user", "content": f"Task: {request.task}\nDate/Time: {reminder_time}\nPriority: {request.priority}"}
            ],
        )

        reminder_message = response.choices[0].message.content

        reminder_entry = {
            "task": request.task,
            "scheduled_for": reminder_time.strftime("%Y-%m-%d %H:%M"),
            "priority": request.priority,
            "reminder": reminder_message,
            "status": "Scheduled"
        }
        reminders_db.append(reminder_entry)

        return {"message": "Reminder created successfully!", "reminder": reminder_entry}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reminders", tags=["Reminders"])
async def get_reminders():
    """Retrieve reminders, flag due soon/overdue"""
    if not reminders_db:
        return {"message": "No reminders available."}
    
    now = datetime.now()
    updated = []
    for r in reminders_db:
        scheduled_time = datetime.strptime(r["scheduled_for"], "%Y-%m-%d %H:%M")
        if now <= scheduled_time <= now + timedelta(minutes=30):
            r["status"] = "⚠️ Due Soon"
        elif scheduled_time < now:
            r["status"] = "⏰ Overdue"
        else:
            r["status"] = "Scheduled"
        updated.append(r)

    return {"reminders": updated}

@app.delete("/reminders/{task}", tags=["Reminders"])
async def delete_reminder(task: str):
    """Delete a reminder by task"""
    global reminders_db
    reminders_db = [r for r in reminders_db if r["task"].lower() != task.lower()]
    return {"message": f"Reminder for '{task}' deleted (if it existed)."}

#Calendar
@app.post("/calendar/add_event", tags=["Calendar"])
async def add_event(request: EventRequest):
    """Add a calendar event (with conflict check)"""
    try:
        event_time = datetime.strptime(request.date_time, "%Y-%m-%d %H:%M")
        end_time = event_time + timedelta(minutes=request.duration_minutes)

        conflicts = []
        for event in calendar_db:
            existing_start = datetime.strptime(event["date_time"], "%Y-%m-%d %H:%M")
            existing_end = existing_start + timedelta(minutes=event["duration_minutes"])
            if event_time < existing_end and end_time > existing_start:
                conflicts.append(event)

        new_event = {
            "title": request.title,
            "date_time": request.date_time,
            "duration_minutes": request.duration_minutes,
            "description": request.description,
            "attendees": request.attendees,
            "conflicts": conflicts
        }
        calendar_db.append(new_event)

        return {
            "message": "Event added successfully!",
            "event": new_event,
            "conflicts_detected": conflicts if conflicts else "No conflicts"
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD HH:MM")

@app.get("/calendar/events", tags=["Calendar"])
async def get_events():
    """Retrieve all events"""
    if not calendar_db:
        return {"message": "No events scheduled."}
    return {"events": calendar_db}

@app.delete("/calendar/events/{title}", tags=["Calendar"])
async def delete_event(title: str):
    """Delete event by title"""
    global calendar_db
    before = len(calendar_db)
    calendar_db = [e for e in calendar_db if e["title"].lower() != title.lower()]
    after = len(calendar_db)
    
    if before == after:
        return {"message": f"No event found with title '{title}'"}
    return {"message": f"Event '{title}' deleted successfully."}


#to run it on terminal sir, use uvicorn ai_app:app --reload, it'll paste a link that youll use on your browser