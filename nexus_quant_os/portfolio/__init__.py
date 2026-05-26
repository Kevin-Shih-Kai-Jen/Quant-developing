"""Portfolio construction & optimisation.

v2.0 新增：
    - WeightSmoother   : EMA 權重平滑 + 最小換手閾值
    - RegimeAllocator  : 政體自適應配置（HMM 先驗混合）
    - PortfolioOptimizer: 雙層投組優化（Risk Parity + Constrained MVO）
"""
