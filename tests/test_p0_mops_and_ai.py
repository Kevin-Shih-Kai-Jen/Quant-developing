import os
import time
import threading
import pytest
from unittest.mock import patch, MagicMock

from nexus_quant_os.alpha_hunter.mops_scraper import MOPSScraper, MOPSRateLimiter
from nexus_quant_os.alpha_hunter.ai_analyst import AIAnalyst


def test_mops_ajax_bypass(mocker):
    """測試繞過 VIEWSTATE，直接 POST ajax 節點。"""
    scraper = MOPSScraper(max_retries=1)
    
    # Mock requests.post
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    # 假裝 MOPS 回傳了一個帶有 <table class="hasBorder"> 的 HTML
    mock_resp.text = '''
    <html>
      <table class="hasBorder">
        <tr><th>日期</th><th>時間</th><th>主旨</th><th>內容</th></tr>
        <tr><td>113/05/10</td><td>15:00:00</td><td>法說會</td><td>本公司召開法說會</td></tr>
      </table>
    </html>
    '''
    mocker.patch("requests.Session.post", return_value=mock_resp)
    
    # Mock MOPSRateLimiter.wait 來加速測試
    mocker.patch("nexus_quant_os.alpha_hunter.mops_scraper.MOPSRateLimiter.wait")
    
    results = scraper.get_company_announcements("2330", 2024, 5)
    assert len(results) == 1
    assert results[0]["subject"] == "法說會"


def test_global_rate_limiter():
    """測試全域速率限制器是否有強制排隊 (用 Thread 模擬 parallel requests)。"""
    import time
    
    call_times = []
    original_sleep = time.sleep
    
    def mock_sleep(secs):
        call_times.append(time.time())
        original_sleep(0.1)  # 縮短等待時間以加速測試，但保留順序與鎖的行為
        
    try:
        import nexus_quant_os.alpha_hunter.mops_scraper as ms
        ms.time.sleep = mock_sleep
        
        def worker():
            MOPSRateLimiter.wait()
            
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert len(call_times) == 5
        for i in range(1, 5):
            assert call_times[i] - call_times[i-1] >= 0.09
    finally:
        ms.time.sleep = original_sleep


def test_pdf_multimodal_delegation(mocker):
    """測試 PDF 多模態處理 (Gemini Vision API)。"""
    analyst = AIAnalyst(api_key="TEST_API_KEY")
    
    test_pdf_path = "/tmp/nexus_test_mock.pdf"
    with open(test_pdf_path, "w") as f:
        f.write("mock pdf")
        
    try:
        mock_upload = mocker.patch("google.generativeai.upload_file")
        mock_file_obj = MagicMock()
        mock_upload.return_value = mock_file_obj
        
        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"summary": "PDF解析成功", "ai_score": 0.8}'
        mock_model.generate_content.return_value = mock_resp
        
        mocker.patch("google.generativeai.GenerativeModel", return_value=mock_model)
        
        analysis = analyst.analyze(ticker="MOCKPDF.TW", mda_text=test_pdf_path)
        
        mock_upload.assert_called_once_with(path=test_pdf_path)
        args = mock_model.generate_content.call_args[0]
        assert args[0][1] == mock_file_obj
        mock_file_obj.delete.assert_called_once()
        
        assert analysis.summary == "PDF解析成功"
        assert analysis.ai_score == 0.8
    finally:
        if os.path.exists(test_pdf_path):
            os.remove(test_pdf_path)


def test_prompt_tw_semantics(mocker):
    """測試在地化語意字典注入 (CRITICAL OVERRIDE)。"""
    analyst = AIAnalyst(api_key="TEST_API_KEY")
    
    mock_post = mocker.patch("requests.Session.post")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"ai_score": 0.5}'}]}}]
    }
    mock_post.return_value = mock_resp
    
    analyst.analyze(ticker="MOCKSEM.TW", mda_text="測試", risk_text="測試", recent_news=["長老吃筍", "亮燈"])
    
    kwargs = mock_post.call_args[1]
    payload = kwargs["json"]
    prompt_text = payload["contents"][0]["parts"][0]["text"]
    
    assert "CRITICAL OVERRIDE: 分析台灣股票時，若遇到『長老吃筍』，請翻譯為『政府八大行庫停損』" in prompt_text
    assert "長老吃筍" in prompt_text
    assert "亮燈" in prompt_text
