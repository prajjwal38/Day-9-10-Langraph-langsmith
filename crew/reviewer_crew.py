from crewai import Agent, Task, Crew, Process

def create_reviewer_crew():
    reviewer = Agent(
        role="Reviewer",
        goal="Critically analyze and refine written content for clarity and accuracy",
        backstory="Expert in editorial review and proofreading"
    )

    task = Task("Review the summary for clarity, grammar, and accuracy", agent=reviewer)

    crew = Crew(
        agents=[reviewer],
        tasks=[task],
        process=Process.sequential
    )

    return crew