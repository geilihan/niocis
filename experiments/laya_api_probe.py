# Probe laya package API for local-checkpoint loading surface.
import inspect
from laya.agent import Agent
from laya.router import Router

print("Agent init:", inspect.signature(Agent.__init__))
print("Router methods:", [n for n in dir(Router) if not n.startswith("_")])
print("Router.load:", inspect.signature(Router.load))
print("Router.predict:", inspect.signature(Router.predict))