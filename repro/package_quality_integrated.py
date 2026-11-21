from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import re

root = Path.cwd()
edition = root/'artical'/'nadir-quality-integrated-v2'
archive = edition/'quality-integrated-v2-source.zip'
paper = ('quality-integrated','quality-integrated-supplement')
for stem, text_name in zip(paper,('main-release','supp-check')):
    log = (edition/f'{stem}.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'Overfull|undefined|Missing character|^!',log,re.M|re.I), stem
    pdf_text = (edition/'review-pages'/f'{text_name}.txt').read_text(encoding='utf-8')
    assert not re.search(r'\?\?|待填|TODO|TBD|E:[\\/]|repro[/\\]',pdf_text,re.I), stem

trace = edition/'数字追溯.csv'
with trace.open(encoding='utf-8-sig',newline='') as file:
    rows = list(csv.DictReader(file))
assert len(rows) >= 50
for row in rows:
    assert (root/row['正式源文件']).is_file(), row
needed = ['README.md','数字追溯.csv','修改清单.md','复核记录.md',
          'numbers.tex','ecgr-table.tex','quality-table.tex','mis-table.tex']
files = [edition/file for file in needed]
files += [edition/f'{stem}{extension}' for stem in paper for extension in ('.tex','.pdf')]
files += sorted((edition/'figures').glob('*.pdf')) + [edition/'figures/ecgr-flow.tex']
files += sorted(file for file in (edition/'evidence').rglob('*') if file.is_file())
scripts = ('build_quality_integrated.py','run_nadir_ecgr.ps1','run_nadir_quality.ps1',
           'nadir_ecgr.m','nadir_ecgr.py','nadir_quality.m','nadir_quality.py',
           'prepare_nadir_quality.py','nadir_pair_test.m','nadir_pair_predict.py',
           'verify_and_plot_nadir_ecgr.py','verify_plot_nadir_quality.py')
with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=6) as package:
    for file in files:
        package.write(file,file.relative_to(edition))
    package.write(root/'reports/项目级证据与版本归属.md','项目级证据与版本归属.md')
    for name in scripts:
        package.write(root/'repro'/name,'repro-reference/'+name)
with ZipFile(archive) as package:
    assert package.testzip() is None
    for stem in paper:
        tex = package.read(stem+'.tex').decode('utf-8')
        for source in re.findall(r'\\input\{([^}]+)\}',tex):
            assert source+'.tex' in package.namelist(), source
        for source in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',tex):
            assert source in package.namelist(), source
    print(f'ZIP PASS: {len(package.namelist())} entries, {len(rows)} numeric trace rows, {archive.stat().st_size:,} bytes')
