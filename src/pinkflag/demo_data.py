"""Add demo guys so the app (roster, wheel, receipts) has something to show.

    python -m pinkflag.demo_data

His messages are analysed by the REAL trained model (no Gemini calls, no quota used).
Running it twice doesn't duplicate anyone.
"""
import sys

from pinkflag.agent.memory import Memory
from pinkflag.agent.tools import Toolbox

DEMO = [
    {"name": "Nour (demo)", "emoji": "\U0001f3a8", "status": "dating",
     "profile": {"works_or_studies": "software engineer", "met_at": "at work",
                 "age": "24", "detail": "likes painting"},
     "his_messages": ["I love you very very much"],
     "notes": [],
     "chat": [("user", 'He texted me "I love you very very much".'),
              ("model", "How long have you been together, and how did that message feel to you?")]},
    {"name": "Youssef", "emoji": "🎧", "status": "situationship",
     "profile": {"works_or_studies": "studies engineering", "met_at": "Instagram",
                 "age": "23", "detail": "DJs on weekends"},
     "his_messages": ["wyd later", "heyyy stranger", "sorry been busy for 3 weeks, miss you tho",
                      "we should hang out sometime idk", "can't tonight, maybe next week",
                      "you looked so good today fr"],
     "notes": ["5 months of talking, no real date yet", "disappears for weeks then texts 'wyd'"],
     "chat": [("user", "youssef texted me 'heyyy stranger' after 3 weeks of nothing 🙄"),
              ("model", "the 3-week ghost then 'heyyy stranger' combo 😮‍💨 that's breadcrumbing, babe. "
                        "he gives you just enough attention to keep you around. what do YOU want from this?")]},
    {"name": "Karim", "emoji": "📚", "status": "talking stage",
     "profile": {"works_or_studies": "med student, 5th year", "met_at": "the library",
                 "age": "25", "detail": "always texts good morning"},
     "his_messages": ["good morning! hope your exam goes well", "text me when you get home safe",
                      "no worries if you're busy tonight, we can reschedule",
                      "i booked the restaurant you mentioned, is 8 okay?",
                      "i'm proud of you for finishing that project"],
     "notes": ["3 weeks of talking", "consistent, plans dates ahead"],
     "chat": [("user", "karim booked the restaurant i mentioned like a week ago 🥹"),
              ("model", "he LISTENED and he PLANNED?? 💚 that's green flag energy, bestie. "
                        "consistency is the most attractive thing a man can do.")]},
    {"name": "Mehdi", "emoji": "🧢", "status": "ex",
     "profile": {"works_or_studies": "works in sales", "met_at": "at work", "age": "27",
                 "detail": "still texts at 2am"},
     "his_messages": ["you'll regret this, trust me", "you're overreacting like always",
                      "nobody else would put up with you", "ANSWER YOUR PHONE RIGHT NOW",
                      "if you really loved me you would come back"],
     "notes": ["broke up 2 months ago", "she blocked him once already"],
     "chat": [("user", "mehdi texted 'you'll regret this, trust me' 😰"),
              ("model", "okay, I'm being serious now. that's a threat, not a feeling. you did nothing wrong. "
                        "save the screenshots, tell someone you trust, and you never have to answer him.")]},
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    memory = Memory()
    existing = {g["name"] for g in memory.guys()}
    tools = Toolbox(memory)
    for d in DEMO:
        if d["name"] in existing:
            print(f"  ⏭️  {d['name']} already there")
            continue
        gid = memory.add_guy(d["name"], d["status"], d["emoji"])
        memory.update_guy(gid, **d["profile"])
        tools.guy_id = gid
        tools.analyze_messages(d["his_messages"])  # labels come from the real model
        for note in d["notes"]:
            memory.remember(note, f"guy-{gid}")
        for role, text in d["chat"]:
            memory.add_message(f"guy-{gid}", role, text)
        print(f"  ✅ added {d['emoji']} {d['name']} ({len(d['his_messages'])} messages analysed)")


if __name__ == "__main__":
    main()
