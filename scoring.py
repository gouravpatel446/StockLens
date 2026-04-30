"""
Fundamental Scoring Module
Evaluates stocks based on key financial metrics and provides a risk/quality score
"""


def fundamental_score(info):
    """
    Calculate a fundamental score for a stock based on key financial metrics.
    
    This function analyzes 4 important financial indicators to provide an overall
    assessment of a stock's fundamental health and quality.
    
    Metrics Analyzed:
    -----------------
    1. PE Ratio (Price-to-Earnings): 
       - Measures if stock is expensive or cheap
       - Lower is generally better (< 25 is good)
    
    2. Debt/Equity Ratio:
       - Shows how much debt a company has compared to equity
       - Lower is better (< 100 means manageable debt)
    
    3. Revenue Growth:
       - Percentage growth in company revenue
       - Higher is better (> 10% is strong growth)
    
    4. ROE (Return on Equity):
       - Shows how efficiently company uses investor money
       - Higher is better (> 15% is good performance)
    
    Parameters:
    -----------
    info : dict
        Dictionary containing stock information with keys:
        - 'trailingPE': Price-to-Earnings ratio
        - 'debtToEquity': Debt to Equity ratio
        - 'revenueGrowth': Revenue growth percentage (decimal, e.g., 0.15 = 15%)
        - 'returnOnEquity': Return on Equity percentage (decimal, e.g., 0.20 = 20%)
    
    Returns:
    --------
    tuple
        (score, label, reasons)
        - score: Integer from 0-4 (number of metrics that passed)
        - label: String like "2/4 — Average"
        - reasons: List of explanations for each metric
    
    Example:
    --------
    >>> info = {
    ...     'trailingPE': 22,
    ...     'debtToEquity': 50,
    ...     'revenueGrowth': 0.12,
    ...     'returnOnEquity': 0.18
    ... }
    >>> score, label, reasons = fundamental_score(info)
    >>> print(label)  # Output: "3/4 — Strong"
    """
    
    # Initialize score counter (starts at 0, max 4)
    score = 0
    
    # List to store feedback for each metric
    reasons = []
    
    # ── Extract Financial Metrics from Info Dictionary ──
    
    # Get PE Ratio (Price-to-Earnings) - measures valuation
    pe_ratio = info.get('trailingPE', None)
    
    # Get Debt-to-Equity ratio - measures financial leverage
    debt_to_equity = info.get('debtToEquity', None)
    
    # Get Revenue Growth - measures sales growth (as decimal)
    revenue_growth = info.get('revenueGrowth', None)
    
    # Get ROE (Return on Equity) - measures profitability on shareholder equity
    return_on_equity = info.get('returnOnEquity', None)
    
    # ── Metric 1: PE Ratio Analysis ──
    
    # Check if PE ratio data is available
    if pe_ratio is not None:
        # Healthy threshold: PE < 25 (stock is reasonably priced)
        if pe_ratio < 25:
            score += 1  # Increment score for good metric
            reasons.append(f'Good - PE Ratio: {round(pe_ratio, 2)} (below 25)')
        else:
            # Stock might be overvalued
            reasons.append(f'Watch - PE Ratio: {round(pe_ratio, 2)} (above 25)')
    else:
        # Data not available
        reasons.append('PE Ratio: Not available')
    
    # ── Metric 2: Debt-to-Equity Ratio Analysis ──
    
    # Check if Debt-to-Equity data is available
    if debt_to_equity is not None:
        # Healthy threshold: Debt/Equity < 100 (manageable debt levels)
        if debt_to_equity < 100:
            score += 1  # Increment score for good metric
            reasons.append(f'Good - Debt/Equity: {round(debt_to_equity, 2)} (below 100)')
        else:
            # Company has high debt levels
            reasons.append(f'Watch - Debt/Equity: {round(debt_to_equity, 2)} (above 100) - High debt')
    else:
        # Data not available
        reasons.append('Debt/Equity: Not available')
    
    # ── Metric 3: Revenue Growth Analysis ──
    
    # Check if Revenue Growth data is available
    if revenue_growth is not None:
        # Strong threshold: Revenue growth > 10% (0.1 in decimal form)
        if revenue_growth > 0.10:
            score += 1  # Increment score for good metric
            # Convert decimal to percentage for display (e.g., 0.15 → 15%)
            growth_percentage = round(revenue_growth * 100, 1)
            reasons.append(f'Good - Revenue Growth: {growth_percentage}% (above 10%)')
        else:
            # Growth is weak or negative
            growth_percentage = round(revenue_growth * 100, 1)
            reasons.append(f'Watch - Revenue Growth: {growth_percentage}% (below 10%)')
    else:
        # Data not available
        reasons.append('Revenue Growth: Not available')
    
    # ── Metric 4: ROE (Return on Equity) Analysis ──
    
    # Check if ROE data is available
    if return_on_equity is not None:
        # Strong threshold: ROE > 15% (0.15 in decimal form)
        if return_on_equity > 0.15:
            score += 1  # Increment score for good metric
            # Convert decimal to percentage for display (e.g., 0.20 → 20%)
            roe_percentage = round(return_on_equity * 100, 1)
            reasons.append(f'Good - ROE: {roe_percentage}% (above 15%)')
        else:
            # ROE is below target
            roe_percentage = round(return_on_equity * 100, 1)
            reasons.append(f'Watch - ROE: {roe_percentage}% (below 15%)')
    else:
        # Data not available
        reasons.append('ROE: Not available')
    
    # ── Generate Overall Label Based on Score ──
    
    # Score: 0-4 points based on how many metrics are "healthy"
    if score >= 3:
        # 3-4 metrics are good: Strong stock
        overall_label = 'Strong'
    elif score >= 2:
        # 2 metrics are good: Average stock
        overall_label = 'Average'
    else:
        # 0-1 metrics are good: Weak stock
        overall_label = 'Weak'
    
    # ── Return Results ──
    
    # Create formatted label showing score and assessment
    formatted_label = f'{score}/4 — {overall_label}'
    
    return score, formatted_label, reasons


def get_score_color(score):
    """
    Get a color indicator based on the fundamental score.
    
    Useful for visualizing the score in the UI.
    
    Parameters:
    -----------
    score : int
        Fundamental score from 0-4
    
    Returns:
    --------
    str
        Color code: 'green' (strong), 'orange' (average), 'red' (weak)
    
    Example:
    --------
    >>> color = get_score_color(3)
    >>> print(color)  # Output: "green"
    """
    
    if score >= 3:
        return 'green'  # Strong score
    elif score >= 2:
        return 'orange'  # Average score
    else:
        return 'red'  # Weak score


def get_score_emoji(score):
    """
    Get an emoji representation of the fundamental score.
    
    Parameters:
    -----------
    score : int
        Fundamental score from 0-4
    
    Returns:
    --------
    str
        Emoji: 'UP' (strong), 'NEUTRAL' (average), 'DOWN' (weak)
    
    Example:
    --------
    >>> emoji = get_score_emoji(2)
    >>> print(emoji)  # Output: "NEUTRAL"
    """
    
    if score >= 3:
        return 'UP'  # Strong
    elif score >= 2:
        return 'NEUTRAL'  # Average
    else:
        return 'DOWN'  # Weak


# ── Quick Reference Guide ──
"""
INTERPRETATION GUIDE:
====================

PE Ratio (Price-to-Earnings):
- Good: < 25 (stock is reasonably priced)
- Warning: > 25 (stock might be expensive)

Debt-to-Equity:
- Good: < 100 (company has manageable debt)
- Warning: > 100 (company has high debt)

Revenue Growth:
- Good: > 10% (strong sales growth)
- Warning: < 10% (slow sales growth)

ROE (Return on Equity):
- Good: > 15% (efficient use of shareholder money)
- Warning: < 15% (less efficient)

OVERALL SCORE:
- 3-4 points: STRONG fundamentals
- 2 points: AVERAGE fundamentals
- 0-1 points: WEAK fundamentals
"""
