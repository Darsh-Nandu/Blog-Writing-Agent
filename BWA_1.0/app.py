from nodes import build_graph

blog_app = build_graph()

out = blog_app.invoke({"topic": "Write a blog on Self Attention", "sections": []})

print(out)