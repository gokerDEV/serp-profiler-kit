import asyncio

async def scraper(browser, url='https://goker.me', file_name=None, ss_file_name=None):
    page = None
    try:
        page = await browser.newPage()
  
        # DESKTOP_EMULATION_METRICS
        # https://github.com/GoogleChrome/lighthouse/blob/main/core/config/constants.js
        viewport_height = 940
        await page.setViewport({'width': 1350, 'height': viewport_height})
        await page.setJavaScriptEnabled(True)  # Ensure JS is enabled

        # Set common headers
        await page.setExtraHTTPHeaders({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'DNT': '1',
        })
        
        await page.goto(url, {'waitUntil': 'load', 'timeout': 5000})
        
        # https://stackoverflow.com/questions/55506935/puppeteer-screenshot-lazy-images-not-working
        if ss_file_name:
            
            # detect the lazy loading
            await page.evaluate(f'window.scrollTo(0, {viewport_height})')
            height = await page.evaluate('() => Math.max(document.body.scrollHeight, document.body.clientHeight)')
            
            vhIncrease = 0
           
            while vhIncrease + viewport_height < height:
                await page.evaluate(f'window.scrollTo(0, {viewport_height})')
                await page.waitFor(300)  
                vhIncrease += viewport_height
                
            await page.setViewport({'width': 1350, 'height': height})
            await page.waitFor(500);
            await page.evaluate('window.scrollTo(0, 0)')
            # Capture the screenshot using the viewport dimensions
            await page.screenshot({
                'path': f'{ss_file_name}.png',
            })
            await page.setViewport({'width': 1350, 'height': viewport_height})
            
        htmlContent = await page.content()
        with open(file_name+'.html', 'w') as f:
            f.write(htmlContent)

        return {'success': True}

    
    except asyncio.TimeoutError:
        raise Exception(f'Timeout exceeded while waiting for page load: {url}')
        
    finally:
        if page:
            try:
                await page.close()
            except Exception as close_err:
                print(f"Warning: Error closing page for {url}: {close_err}")
            finally:
                return {'success': False}


