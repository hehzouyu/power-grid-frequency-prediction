"""Assemble paper assets from approved CSV files, without simulation or model execution."""
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artical/nadir-quality-integrated-v2"
EC = ROOT / "repro/out/nadir-ecgr/confirmation-v1"
QC = ROOT / "repro/out/nadir-quality/confirm"
NAMES = dict(endpoint="端点",initial_rate="初始斜率",quadratic="二次拟合",
             fitted_segments="分段拟合",ecgr="ECGR",gate="门控")
CONDS = dict(clean="干净",white40="白噪40 dB",white30="白噪30 dB",white20="白噪20 dB",
             ar30="AR(1) 30 dB",jump="突变")
SOURCES = []
MACROS = {}

def trace(name,value,source,key,field,script,fmt=".2f"):
    value = format(value,fmt) if isinstance(value,(float,np.floating)) else str(value)
    MACROS[name] = value
    SOURCES.append([name,value,str(source.relative_to(ROOT)),key,field,script])

def values():
    e=EC/"pair-results.csv"; ep=pd.read_csv(e).set_index(["Condition","Feature"])
    ec=EC/"paired-contrasts.csv"; ce=pd.read_csv(ec).set_index(["Condition","Comparator"])
    q=QC/"metrics.csv"; qp=pd.read_csv(q).set_index(["Kind","Condition","Feature"])
    c=QC/"main-contrasts.csv"; contrasts=pd.read_csv(c).set_index("Kind")
    m=QC/"mis-selection.csv"; mis=pd.read_csv(m).set_index(["Kind","Condition"])
    def get(name,file,frame,key,field,script,fmt=".2f",scale=1):
        trace(name,float(frame.loc[key,field])*scale,file,str(key),field+("" if scale==1 else f" × {scale}"),script,fmt)
    eg="repro/nadir_ecgr.py"; qg="repro/nadir_quality.py"
    for name,feature in (("E","ecgr"),("Q","quadratic")):
        key=("clean",feature)
        for code,field in (("Mae","MAE_mHz"),("Tail","P95_mHz")):
            get(name+"clean"+code,e,ep,key,field,eg)
        row=ep.loc[key]
        trace(name+"dir",f"{int(row.CorrectDirections)}/{int(row.DirectionTrials)}",e,str(key),
              "CorrectDirections/DirectionTrials",eg)
    for prefix,key in (("E",("clean","quadratic")),("Ewhite",("white30","endpoint"))):
        for name,col in (("change","ChangeMAE_mHz"),("low","Low95_mHz"),("high","High95_mHz")):
            suffix=name if prefix=="E" else name.capitalize()
            get(prefix+suffix,ec,ce,key,col,eg)
    for feat,name in (("ecgr","EwhiteTail"),("endpoint","EndWhiteTail")):
        get(name,e,ep,("white30",feat),"P95_mHz",eg)
    a=EC/"acceptance.csv"
    row=pd.read_csv(a).set_index("Criterion")
    get("EwhiteLimit",a,row,"pair white30 P95_mHz","Threshold",eg)
    for prefix,folder,pairfield in (("E",EC,"PhysicalPairs"),("P",QC,None)):
        file=folder/"pair-index.csv"; rows=pd.read_csv(file)
        for name,v,field in (("refs",rows.ReferenceID.nunique(),"nunique(ReferenceID)"),
                ("eligible",sum(rows.Status=="primary_eligible"),"count(Status=primary_eligible)"),
                ("pairs" if prefix=="E" else "count",len(rows),"row count")):
            trace(prefix+name,v,file,"全部候选",field,"repro/nadir_pair_test.m")
    s=QC/"single-index.csv"
    trace("Srefs",sum(pd.read_csv(s).Status=="ok"),s,"Status=ok","计数","repro/nadir_quality.m")
    es=EC/"freeze.json"; fs=QC.parent/"frozen-selector.json"
    trace("ELambda",json.loads(es.read_text())["lambda"],es,"冻结参数","lambda",eg)
    trace("ECubicMoment",json.loads(es.read_text())["training_cubic_second_moment_hz2"]*1e5,
          es,"冻结参数（显示值乘10^-5）","training_cubic_second_moment_hz2",eg,".13f")
    for n,k in (("GateClean","clean"),("GateFallback","fallback")):
        trace(n,json.loads(fs.read_text())["rule"][k],fs,"rule",k,qg,".4g")
    for prefix,key in (("S","single"),("P","pair")):
        for suffix,col,scale in [("base","BaselineMAE_mHz",1),("gate","GateMAE_mHz",1),
                ("gain","RelativeChange",-100),("lo","High95",-100),("upper" if prefix=="P" else "hi","Low95",-100)]:
            get(prefix+suffix,c,contrasts,key,col,qg,scale=scale)
    for n,kind,cond,feat,col in [
        ("GateWhiteTail","pair","white30","gate","P95_mHz"),("GateSetEcgrTail","pair","white30","ecgr","P95_mHz"),
        ("GateQuadSingle","single","white30","quadratic","MAE_mHz"),("GateWhiteSingle","single","white30","gate","MAE_mHz"),
        ("JumpEnd","single","jump","endpoint","MAE_mHz"),("JumpGate","single","jump","gate","MAE_mHz")]:
        get(n,q,qp,(kind,cond,feat),col,qg)
    aq=QC/"acceptance.csv"
    get("GateWhiteLimit",aq,pd.read_csv(aq).set_index(["Kind","Criterion"]),("pair","white30 P95 vs endpoint"),"Limit",qg)
    jm=QC/"selection-confusion.csv"
    j=pd.read_csv(jm)
    trace("JumpSwitch",int(j.loc[j.Condition.eq("jump")&j.Selected.eq("quadratic"),"N"].sum()),jm,"jump and quadratic","sum(N)",qg)
    for k,v in (("single","Single"),("pair","Pair")):
        for n,f,s in [("Mis","MisselectRate",100),("Loss","MeanOracleLoss_mHz",1)]:
            get("Jump"+v+n,m,mis,(k,"jump"),f,qg,scale=s)

def tables():
    specs=[
        ("ecgr-table",EC/"pair-results.csv","llrrrr",["条件","模型","MAE","RMSE","P95","方向判对/总数"]),
        ("quality-table",QC/"metrics.csv","lllrrrr",["任务","条件","模型","MAE","RMSE","P95","方向判对/总数"]),
        ("mis-table",QC/"mis-selection.csv","llrrr",["任务","条件","误选率/\\%","平均损失","P95损失"]),
    ]
    for name,file,cols,header in specs:
        frame=pd.read_csv(file)
        first=" & ".join(header)+r" \\"
        lines=[r"\small",r"\begin{longtable}{"+cols+"}",r"\toprule",first,r"\midrule",r"\endhead"]
        for row in frame.to_dict("records"):
            part=[]
            if "Kind" in row: part.append({"single":"单场景","pair":"配对"}[row["Kind"]])
            part.append(CONDS[row["Condition"]])
            if "Feature" in row:
                part.append(NAMES[row["Feature"]])
                part.extend(f"{row[k]:.2f}" for k in ("MAE_mHz","RMSE_mHz","P95_mHz"))
                d=row.get("DirectionN",row.get("DirectionTrials",0))
                hits=row.get("DirectionHits",row.get("CorrectDirections",0))
                part.append(f"{int(hits)}/{int(d)}" if d else "—")
            else:
                part.extend([f'{100*row["MisselectRate"]:.2f}',f'{row["MeanOracleLoss_mHz"]:.2f}',f'{row["P95OracleLoss_mHz"]:.2f}'])
            lines.append(" & ".join(part)+r" \\")
        lines += [r"\bottomrule",r"\end{longtable}",r"\normalsize 各表误差及损失单位为mHz。"]
        (DEST/f"{name}.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")

def figures():
    figures=DEST/"figures"
    for f in [EC/"ecgr-paired-contrasts.pdf", EC/"ecgr-pair-errors.pdf", *sorted(QC.glob("quality-*.pdf"))]:
        copyfile(f,figures/f.name)
    source=QC/"predictions.csv"
    strategies=pd.read_csv(source)
    selected=strategies.query("Kind=='single' and Condition=='jump' and Selected=='quadratic'")
    assert len(selected)==2
    idx=np.arange(len(selected))
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"pdf.fonttype":42})
    fig,ax=plt.subplots(1,2,figsize=(8,2.75),layout="constrained")
    knobs=[MACROS["GateClean"], MACROS["GateFallback"]]
    ax[0].scatter(selected.PreQuality,idx,label="扰动前",color="#4c6c86")
    ax[0].scatter(selected.WindowQuality,idx,label="窗内",color="#bf6442")
    for k in knobs:
        ax[0].axvline(float(k),color="#777",ls="--",lw=.8)
    ax[0].set(xlabel="质量代理",yticks=idx,yticklabels=[f"{r.Entity} / 复本 {r.Rep}" for r in selected.itertuples()])
    ax[0].legend(frameon=False,fontsize=8)
    for j,(name,caption,col) in enumerate((("endpoint","固定端点","#4c6c86"),("gate","门控","#bf6442"))):
        ax[1].barh(idx+(j-.5)*.29,1000*abs(selected[name]-selected.Target_Hz),height=.27,label=caption,color=col)
    ax[1].set(yticks=idx,yticklabels=[],xlabel="绝对误差 / mHz")
    ax[1].legend(frameon=False,fontsize=8)
    fig.savefig(figures/"jump-failures.pdf")
    plt.close(fig)
    SOURCES.append(["突变案例图","两例",str(source.relative_to(ROOT)),"single/jump/Selected=quadratic",
                    "PreQuality,WindowQuality, abs(prediction-Target_Hz)×1000","repro/build_quality_integrated.py"])

def build():
    for name in ("figures","evidence"):
        (DEST/name).mkdir(parents=True,exist_ok=True)
    values(); tables(); figures()
    (DEST/"numbers.tex").write_text("\n".join(r"\newcommand{\%s}{%s}"%(n,v) for n,v in MACROS.items())+"\n",encoding="utf-8")
    for folder,names,sub in [(EC,("pair-results.csv","paired-contrasts.csv","single-results.csv","pair-index.csv",
                                  "acceptance.csv","freeze.json","independent-recheck.log","method-protocol.json","pair-status-counts.csv",
                                  "excluded-and-failed.csv","member-predictions.csv","pair-predictions.csv","protocol.json"),"ecgr"),
          (QC,("metrics.csv","main-contrasts.csv","mis-selection.csv","selection-confusion.csv","pair-selection-confusion.csv",
               "status-counts.csv","pair-index.csv","excluded-and-failed.csv","single-index.csv","worst-single-jump.csv",
               "worst-pair-jump.csv","acceptance.csv","dual-study-readback.log","predictions.csv","pair-predictions.csv",
               "protocol.json","confirmation.log"),"quality")]:
        target=DEST/"evidence"/sub;target.mkdir(exist_ok=True)
        for name in names: copyfile(folder/name,target/name)
    copyfile(QC.parent/"frozen-selector.json",DEST/"evidence/quality/frozen-selector.json")
    for file in ("ecgr-table","quality-table","mis-table"):
        src={"ecgr-table":EC/"pair-results.csv","quality-table":QC/"metrics.csv","mis-table":QC/"mis-selection.csv"}[file]
        SOURCES.append([file,"逐行",""+str(src.relative_to(ROOT)),"全部",",".join(pd.read_csv(src,nrows=0).columns),
                        "repro/build_quality_integrated.py"])
    for study, folder, status_file, script in (
        ("ECGR", EC, "pair-status-counts.csv", "repro/nadir_ecgr.m"),
        ("门控", QC, "status-counts.csv", "repro/nadir_quality.m"),
    ):
        status = pd.read_csv(folder / status_file)
        counts = status.groupby("Status")["Count"].sum()
        value = f'并列{counts["true_tie"]}，小差{counts["below_minimum_gap"]}'
        SOURCES.append([study+"样本流转", value, str((folder/status_file).relative_to(ROOT)),
                        "true_tie/below_minimum_gap", "Count 按状态求和", script])
        SOURCES.append([study+"配对设计", "0.10 mHz及两调差带", str((folder/"protocol.json").relative_to(ROOT)),
                        "冻结配对参数","endpoint_tolerance_hz ×1000 / r_bands / candidate_factors",script])
    fields=["位置或变量","显示值","正式源文件","筛选键","计算字段","生成脚本"]
    pd.DataFrame(SOURCES,columns=fields).to_csv(DEST/"数字追溯.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__":
    build()
