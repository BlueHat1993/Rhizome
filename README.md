# Rhizomatic Network Simulator

**A minimal simulation of rhizomatic behavior through LLM-based node interactions in a dynamic graph.**

## 📖 Concept

A *rhizome*, as conceptualized in post-structuralist theory, is an assemblage that allows any element to connect to any other, with no hierarchy, linearity, or fixed entry point. This idea, developed by French theorists **Gilles Deleuze** and **Félix Guattari** in their work on *schizoanalysis*, serves as the inspiration for this experimental project.

This project presents an extremely basic, but functional, representation of a rhizome using a graph-based model. Each node is a conceptual actor that can respond and trigger interactions with neighboring nodes, simulating decentralized and dynamic flows of communication.

## 🔧 How It Works

- The system is modeled as a **graph**, where:
  - Each **node** has the following attributes:
    - `label` / `id`: Unique identifier for the node.
    - `weight`: Currently unused, but reserved for future extensions (e.g., influence, priority).
    - `concept`: Defines the role of the node — one of:
      - `tyrant`
      - `analyst`
      - `mathematician`
      - `philosopher`
    - `response`: Textual content generated via a **LangChain LLM**.

- The simulation begins at **Node 0**:
  - A **user prompt** triggers the first LLM response at Node 0.
  - Neighboring nodes react based on the initiating node’s response.
  - This cascade continues, as each node responds to its neighbors, creating a decentralized web of interactions.

- All activity is logged and visualized through a **Streamlit dashboard**, providing a real-time interface for observing the network’s behavior.

## 🎯 Purpose

This project is an early-stage prototype to explore:

- How basic rhizomatic principles can be modeled computationally.
- How decentralized nodes react to LLM-generated stimuli.
- The potential of combining **LLMs**, **graph structures**, and **interface logging** for experimental post-structuralist simulations.

## 🚀 Future Directions

- Incorporate more complex node roles and behaviors.
- Utilize `weight` in node influence or decision-making logic.
- Introduce multi-agent LLM interactions with distinct personalities.
- Expand graph structure dynamically during runtime.
- Add memory and historical context to each node’s response generation.

## 📷 Interface

The Streamlit dashboard logs each message, response, and node interaction — giving users a transparent view of the evolving rhizomatic communication network.

## 📦 Dependencies

- [LangChain](https://github.com/langchain-ai/langchain) — For LLM interfacing  
- [Streamlit](https://streamlit.io/) — Dashboard and logging interface  
- [NetworkX](https://networkx.org/) — (Assumed) for graph structure and node management

## 🧠 Inspiration

> *"A rhizome ceaselessly establishes connections between semiotic chains, organizations of power, and circumstances relative to the arts, sciences, and social struggles."*  
> — Deleuze & Guattari, *A Thousand Plateaus*
