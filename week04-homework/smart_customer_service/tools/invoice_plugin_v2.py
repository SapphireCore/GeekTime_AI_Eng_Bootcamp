# tools/invoice_plugin_v2.py
from langchain.tools import tool
import time

@tool
def generate_invoice_tool(order_id: str) -> dict:
    """
    Invoice plugin v2 (hot-updated version).
    """
    print(f"--- [Plugin:v2] generate_invoice_tool: order_id={order_id} ---")
    time.sleep(0.1)
    if order_id and order_id.upper().startswith("SN"):
        # Changed URL format to demonstrate hot update effect
        invoice_url = f"https://example.com/new-invoice/invoices-v2/{order_id}.pdf"
        return {
            "success": True,
            "order_id": order_id,
            "invoice_url": invoice_url,
            "message": f"[v2] 发票已生成，下载链接：{invoice_url}"
        }
    return {"success": False, "order_id": order_id, "error": "[v2] 无效的订单号，无法生成发票。"}
