from crewai import Agent, Task, Crew, Process

def create_research_crew():
    researcher = Agent(
        role="Researcher",
        goal="Find accurate and recent information about the given topic",
        backstory="Expert in online research and data gathering"
    )

    writer = Agent(
        role="Writer",
        goal="Summarize the research into a concise, clear answer",
        backstory="Professional technical writer"
    )

    task1 = Task("Research the topic and gather 3 recent insights", agent=researcher)
    task2 = Task("Summarize the research into a well-written paragraph", agent=writer)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential
    )

    return crew
