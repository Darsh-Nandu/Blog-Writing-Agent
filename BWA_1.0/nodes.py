from custom_objects import State, Plan, Task
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from pathlib import Path
from langgraph.graph import StateGraph, START, END


llm = ChatOllama(model="tinyllama")


# This node breaks down the topic into sub-topics
def orchestrator_node (state: State) -> dict:
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "Create a blog plan with 5-7 sections on the following topic."
                )
            ),
            HumanMessage(content=(f"Tpoic: {state['topic']}"))
        ]
    )
    return {"plan": plan}



# This node help allocate the sub-topics to different worker nodes 
def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state['topic'], "plan": state["plan"] }) for task in state["plan"].tasks]



# This node creates some text/content for each sub-topic
def worker_node(payload: dict) -> dict:

    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']

    blog_title = plan.blog_title

    section_md = llm.invoke([
        SystemMessage(content="Write one clean markdown section."),
        HumanMessage(
            content=(
                f"Blog: {blog_title}\n"
                f"Topic: {topic}\n\n"
                f"Section: {task.title}\n"
                f"Brief: {task.brief}\n\n"
                "Return onlt the section content in Markdown."
            )
        )
    ]).content.strip()

    return {"sections": [section_md]}



# This node helps to concatinate all the sub sections to make a one unified blog
def reducer_node(state: State) -> dict:

    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()
    final_md = f"# {title}\n\n{body}\n"
    prompt = f"""Can you clean the blog given bellow - \n" 
    Blog: {final_md}
    """
    ans = llm.invoke(prompt)
    print(ans)

    # Save to file
    filename = title.lower().replace(" ","_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}


# Buling the Graph
def build_graph():
    graph = StateGraph(State)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("worker", worker_node)
    graph.add_node("reducer", reducer_node)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", fanout, ["worker"])
    graph.add_edge("worker", "reducer")
    graph.add_edge("reducer", END)

    app = graph.compile()

    return app
