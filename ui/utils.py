"""
UI utility functions for Research Weaver
"""


def calculate_token_cost(input_tokens: int, output_tokens: int) -> dict:
    """
    Calculate cost based on token usage
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Dictionary with cost breakdown and total
    """
    input_cost = input_tokens / 1_000_000 * 1.0  # 1 yuan per million input tokens
    output_cost = output_tokens / 1_000_000 * 4.0  # 4 yuan per million output tokens
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }


def format_cost_display(total_cost: float) -> str:
    """
    Format cost for display with appropriate precision
    
    Args:
        total_cost: Total cost in yuan
        
    Returns:
        Formatted cost string
    """
    if total_cost < 0.01:
        return f"¥{total_cost:.4f}"
    elif total_cost < 1:
        return f"¥{total_cost:.3f}"
    else:
        return f"¥{total_cost:.2f}"


def format_token_display(input_tokens: int, output_tokens: int) -> str:
    """
    Format token counts for display with K suffix when appropriate
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Formatted token display string
    """
    input_k = input_tokens / 1000 if input_tokens > 1000 else input_tokens
    output_k = output_tokens / 1000 if output_tokens > 1000 else output_tokens
    
    if input_tokens > 1000:
        return f"{input_k:.1f}K↑/{output_k:.1f}K↓"
    else:
        return f"{int(input_k)}↑/{int(output_k)}↓"


def get_status_emoji(status: str) -> str:
    """
    Get emoji for status display
    
    Args:
        status: Status string
        
    Returns:
        Emoji string for the status
    """
    status_emojis = {
        "idle": "⏸️",
        "initializing": "🚀", 
        "researching": "🔬",
        "searching": "🔍",
        "reading": "📖",
        "thinking": "🧠",
        "processing": "🔄",
        "synthesizing": "🧩",
        "completed": "✅",
        "error": "❌"
    }
    return status_emojis.get(status, "❓")


def format_cost_breakdown(input_tokens: int, output_tokens: int) -> str:
    """
    Format detailed cost breakdown for display
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Multi-line formatted cost breakdown string
    """
    costs = calculate_token_cost(input_tokens, output_tokens)
    
    breakdown = "**Cost Breakdown:**\n"
    breakdown += f"  - Input: {input_tokens:,} tokens × ¥1/M = ¥{costs['input_cost']:.4f}\n"
    breakdown += f"  - Output: {output_tokens:,} tokens × ¥4/M = ¥{costs['output_cost']:.4f}\n"
    breakdown += f"  - **Total: ¥{costs['total_cost']:.4f}**"
    
    return breakdown