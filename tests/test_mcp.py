from mcp_server import server as mcp_server


def test_mcp_tools_wrap_the_registry():
    out = mcp_server.get_order_status("1234")
    assert out["status"] == "packed"

    faq = mcp_server.search_faq("delivery time")
    assert faq["matches"]

    refund = mcp_server.check_refund_policy()
    assert "refund" in refund["policy"].lower()

    esc = mcp_server.escalate_to_human("customer upset")
    assert "human" in esc["ack"].lower()
