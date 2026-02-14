# 🤖 Blog Generator Agent

A streamlined implementation of an **Orchestrator-Worker** design pattern for automated content creation. This workflow transforms a single prompt into a structured, multi-section blog post using parallel processing.

## 🚀 How it Works

The system follows a classic **Fan-Out/Fan-In** architectural pattern to ensure efficiency and depth:

1.  **Input:** The user provides a high-level blog topic via `app.py`.
2.  **Orchestrator Node:** This node analyzes the topic and decomposes it into a structured plan consisting of multiple subtopics.
3.  **Fan-Out (Parallel Workers):** The system triggers multiple **Worker Nodes** simultaneously. Each worker is responsible for generating detailed content for its specific subtopic.
4.  **Reducer Node (Fan-In):** Once all workers complete their tasks, the Reducer node aggregates the fragments, ensures flow, and compiles the final blog.



---

## 📁 Project Structure

| File | Description |
| :--- | :--- |
| **app.py** | The entry point of the application. Used to initialize the agent and pass the user's topic. |
| **nodes.py** | The core logic. Contains the Python functions for the Orchestrator, Workers, and Reducer, as well as the graph definition. |
| **custom_objects.py** | Defines the "Blueprints" of the system, including the **Agent State**, the **Plan** schema, and **Task** formats. |

---

## 🛠️ Key Features

* **Parallel Execution:** Uses fan-out logic to generate multiple sections of a blog at once, significantly reducing total latency.
* **State Management:** Tracks the progress of the blog through a centralized state object, ensuring data consistency between nodes.
* **Structured Planning:** The Orchestrator ensures the blog follows a logical flow before any writing begins.

---

## 🚦 Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/repository-name.git](https://github.com/your-username/repository-name.git)

2. **Install dependencies:**
  (Add your specific install command here, e.g., pip install -r requirements.txt)

4. Run the generator: run the command "python app.py".
