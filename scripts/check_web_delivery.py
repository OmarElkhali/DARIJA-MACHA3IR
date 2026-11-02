"""Vérification HTTP et navigateur headless : aucun contrôle de l’ordinateur ni notebook exécuté."""
from pathlib import Path
import json
import os
import urllib.request
import urllib.error
from playwright.sync_api import sync_playwright

PROJECT=Path(__file__).resolve().parents[1]
URL=os.environ.get('DARIJA_TEST_URL','http://127.0.0.1:8765')
report={}
with urllib.request.urlopen(URL+'/api/status') as response:
    report['engine']=json.load(response)
request=urllib.request.Request(URL+'/api/predict',data=json.dumps({'text':'had lfilm 3jebni bzzaf'}).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=180) as response:
    report['prediction']=json.load(response)
assert len(report['prediction']['results'])==6
assert len([r for r in report['prediction']['results'] if r['status']=='ok'])==5
bad=urllib.request.Request(URL+'/api/predict',data=b'{"text":""}',headers={'Content-Type':'application/json'})
try:
    urllib.request.urlopen(bad)
except urllib.error.HTTPError as error:
    assert error.code==400
else:
    raise AssertionError('Phrase vide acceptée.')
with sync_playwright() as playwright:
    browser=playwright.chromium.launch(headless=True,executable_path=r'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe')
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(URL,wait_until='networkidle')
    assert page.locator('#global-error').is_hidden()
    assert page.locator('#results-table tr').count()==6
    assert page.locator('#variants-table tr').count()==33
    page.locator('[data-family="E"]').click()
    assert 'DarijaBERT-Arabizi' in page.locator('#method-panel').inner_text()
    page.locator('[data-analysis="cost"]').click()
    assert page.locator('#cost-chart img').count()==1
    page.locator('[data-analysis="robustness"]').click()
    page.select_option('#transformation','without_emojis')
    page.locator('.all-results summary').click()
    page.locator('#family-filter').select_option('E')
    assert page.locator('#variants-table tr').count()==4
    page.locator('#family-filter').select_option('all')
    page.locator('#phrase').fill('had lfilm 3jebni bzzaf')
    page.locator('.predict-button').click()
    page.wait_for_function("!document.querySelector('.predict-button').disabled",timeout=180000)
    assert page.locator('.prediction').count()==6
    assert page.locator('.prediction-error').count()==1
    page.locator('#accueil').scroll_into_view_if_needed()
    page.screenshot(path=str(PROJECT/'research/site_desktop.png'),full_page=False)
    page.set_viewport_size({'width':390,'height':844})
    page.goto(URL,wait_until='networkidle')
    overflow=page.evaluate("[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1).slice(0,25).map(e=>({tag:e.tagName,cls:e.className,right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width}))")
    (PROJECT/'research/mobile_overflow.json').write_text(json.dumps(overflow),encoding='utf-8')
    page.screenshot(path=str(PROJECT/'research/site_mobile.png'),full_page=False)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(PROJECT/'research/site_mobile.png'),full_page=False)
    assert not errors,errors
    report['browser']={'desktop':True,'mobile':True,'js_errors':errors,'real_predictions':5,'missing_weight':'E_arabizi_s42'}
    browser.close()
(PROJECT/'research/web_delivery_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Site vérifié : données, filtres, navigation, mobile, API et cinq prédictions réelles. E indisponible explicitement.')
