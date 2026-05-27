"""
nexus_quant_os/advisor — AI Financial Advisor Agent
====================================================

Two implementations:
- ``advisor_v2.GeminiAdvisor``: Standalone Gemini 2.5 Flash (recommended)
- ``agent.NexusAdvisor``: Google Antigravity SDK (requires extra deps)
"""

# Import the standalone version by default (no heavy SDK dependency)
try:
    from nexus_quant_os.advisor.advisor_v2 import GeminiAdvisor
    __all__ = ["GeminiAdvisor"]
except ImportError:
    __all__ = []
