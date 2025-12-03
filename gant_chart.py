import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# Helper: add weeks
def add_weeks(date_str, weeks):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return (date + timedelta(weeks=weeks)).strftime("%Y-%m-%d")

# --- Project start: Week 1 of Spring 2026 semester ---
start = "2026-01-12"  # Adjust this to your actual semester start date

tasks = {
    # PHASE 1: Foundation (Weeks 1-4)
    "EnvSetup":          ("Environment Setup (Node.js, Python, PostgreSQL/PostGIS)", start, 1),
    "DBSchema":          ("Database Schema Design & Setup",                          None,  2),
    "FrontendBase":      ("Complete Frontend Core UI Components",                    None,  3),
    
    # PHASE 2: Backend & Data Pipeline (Weeks 2-7)
    "BackendAPI":        ("Backend API Development (User, Auth, Incidents)",         None,  3),
    "ScraperSetup":      ("Integrate & Test Existing Scrapers",                      None,  2),
    "NLPPipeline":       ("NLP Pipeline (Multilingual Classification & NER)",        None,  4),
    
    # PHASE 3: Verification & ML (Weeks 5-10)
    "VerificationSys":   ("Report Verification & Credibility Scoring System",        None,  3),
    "MLHotspot":         ("ML Hotspot Prediction Model (K-Means, Logistic Reg)",     None,  3),
    "DataIngestion":     ("Data Ingestion Service (Queue-based Processing)",         None,  2),
    
    # PHASE 4: Routing & Safety Integration (Weeks 8-13)
    "RoutingAlgo":       ("Evaluate & Implement Safety-Weighted Routing (OSRM)",     None,  3),
    "SafetyScoring":     ("Safety Score Calculation & Caching System",               None,  2),
    "RouteIntegration":  ("Integrate Safety Scores into Route API",                  None,  2),
    
    # PHASE 5: Integration & Features (Weeks 11-16)
    "FrontendBackend":   ("Connect Frontend to Backend APIs",                        None,  3),
    "MapVisualization":  ("Safety Heatmap & Interactive Map UI (Mapbox)",            None,  2),
    "CommunityModule":   ("Community Groups & Auto-Post System",                     None,  2),
    "NotificationSys":   ("Push Notifications (FCM) & Alert System",                 None,  2),
    "AdminDashboard":    ("Admin Dashboard (Report Moderation & Analytics)",         None,  2),
    
    # PHASE 6: Testing & Deployment (Weeks 15-18)
    "Integration":       ("End-to-End Integration Testing",                          None,  2),
    "UserTesting":       ("User Acceptance Testing & Bug Fixes",                     None,  2),
    "Performance":       ("Performance Optimization & Load Testing",                  None,  1),
    "Deployment":        ("Cloud Deployment & CI/CD Setup",                          None,  1),
    
    # PHASE 7: Documentation & Presentation (Weeks 17-20)
    "FinalReport":       ("Final Report Writing & Documentation",                    None,  3),
    "VideoDemo":         ("Demo Video & Presentation Materials",                     None,  1),
    "PresentationPrep":  ("Final Presentation Rehearsal",                            None,  1),
}

# --- Dependencies (Logical flow based on your architecture) ---
deps = [
    # Foundation dependencies
    ("EnvSetup", "DBSchema"),
    ("EnvSetup", "FrontendBase"),
    ("EnvSetup", "BackendAPI"),
    
    # Backend setup
    ("DBSchema", "BackendAPI"),
    ("BackendAPI", "ScraperSetup"),
    
    # Data pipeline
    ("ScraperSetup", "NLPPipeline"),
    ("NLPPipeline", "VerificationSys"),
    ("BackendAPI", "DataIngestion"),
    ("NLPPipeline", "DataIngestion"),
    
    # ML and routing
    ("VerificationSys", "MLHotspot"),
    ("BackendAPI", "RoutingAlgo"),
    ("MLHotspot", "SafetyScoring"),
    ("RoutingAlgo", "SafetyScoring"),
    ("SafetyScoring", "RouteIntegration"),
    
    # Integration phase
    ("FrontendBase", "FrontendBackend"),
    ("BackendAPI", "FrontendBackend"),
    ("RouteIntegration", "FrontendBackend"),
    ("FrontendBackend", "MapVisualization"),
    ("DataIngestion", "CommunityModule"),
    ("FrontendBackend", "NotificationSys"),
    ("BackendAPI", "AdminDashboard"),
    
    # Testing phase
    ("MapVisualization", "Integration"),
    ("CommunityModule", "Integration"),
    ("NotificationSys", "Integration"),
    ("AdminDashboard", "Integration"),
    ("Integration", "UserTesting"),
    ("UserTesting", "Performance"),
    ("Performance", "Deployment"),
    
    # Final deliverables
    ("Integration", "FinalReport"),
    ("Deployment", "VideoDemo"),
    ("VideoDemo", "PresentationPrep"),
]

# --- Resolve task start dates based on dependencies ---
start_dates = {"EnvSetup": start}

def resolve(task):
    if task in start_dates:
        return start_dates[task]

    # find dependencies that lead to this task
    preds = [a for (a, b) in deps if b == task]

    if preds:
        # finish times of all predecessors
        pred_finishes = []
        for p in preds:
            pred_start = resolve(p)
            p_weeks = tasks[p][2]
            pred_finish = add_weeks(pred_start, p_weeks)
            pred_finishes.append(pred_finish)

        # earliest possible start = max finish among preds
        start_dates[task] = max(pred_finishes)
    else:
        # No dependencies → start immediately
        start_dates[task] = start

    return start_dates[task]

# compute dates for all tasks
df_rows = []
for key, (name, manual_start, dur_weeks) in tasks.items():
    s = resolve(key)
    e = add_weeks(s, dur_weeks)
    df_rows.append(dict(Task=name, Start=s, Finish=e, Duration=f"{dur_weeks}w"))

df = pd.DataFrame(df_rows)

# Calculate week numbers from start
df['StartWeek'] = df['Start'].apply(lambda x: 
    (datetime.strptime(x, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days // 7 + 1
)
df['EndWeek'] = df['Finish'].apply(lambda x: 
    (datetime.strptime(x, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days // 7
)

# --- Create Gantt chart with text INSIDE bars ---
fig = px.timeline(
    df, 
    x_start="Start", 
    x_end="Finish", 
    y="Task",
    title="SafeKarachi FYP - 20 Week Development Timeline",
    color="Duration",
    text="Task"  # Show task names
)

# Update text to be INSIDE the bars with white color for visibility
fig.update_traces(
    textposition="inside",
    insidetextanchor="middle",
    textfont=dict(size=10, color="white", family="Arial Black"),
    marker=dict(line=dict(width=1, color='DarkSlateGrey')),
    cliponaxis=False  # Prevent text from being clipped
)

fig.update_yaxes(
    autorange="reversed", 
    title="Tasks"
)

fig.update_xaxes(title="Timeline")

fig.update_layout(
    height=1100,
    width=1600,
    showlegend=True,
    font=dict(size=11),
    uniformtext=dict(minsize=8, mode='hide')  # Hide text if it doesn't fit
)

fig.show()

# Print summary
print("\n=== SAFEKARACHI FYP TIMELINE SUMMARY ===\n")
print(f"Project Duration: 20 weeks ({start} to {df['Finish'].max()})\n")
print("Key Milestones:")
print(f"  Week 4:  Frontend foundation complete")
print(f"  Week 7:  Backend APIs & NLP pipeline ready")
print(f"  Week 10: Verification & ML models operational")
print(f"  Week 13: Safety-weighted routing implemented")
print(f"  Week 16: Full system integration complete")
print(f"  Week 18: Testing & deployment done")
print(f"  Week 20: Final presentation ready")
print("\n" + "="*45 + "\n")

# Print detailed schedule
print("DETAILED TASK SCHEDULE:\n")
for _, row in df.iterrows():
    print(f"Week {row['StartWeek']:2d}-{row['EndWeek']:2d}: {row['Task']}")