
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor

APP_VERSION = "1.0"
APP_DIR = Path(__file__).resolve().parent

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DIMENSIONS = ["Sincerity","Excitement","Competence","Sophistication","Ruggedness"]

MAPPING_FILE = APP_DIR/"Aaker1997_42traits_15facets_5dimensions_mapping.csv"
REVIEW_FILE = APP_DIR/"03A_eligibility_review_final.csv"

FONT_FILES = {
    "review": APP_DIR/"03B_FontCLIP_logo_review_Aonly.csv",
    "trait": APP_DIR/"03B_FontCLIP_trait_raw_Aonly.csv",
    "facet": APP_DIR/"03B_FontCLIP_facet_raw_Aonly.csv",
    "dim": APP_DIR/"03B_FontCLIP_5D_raw_Aonly.csv",
    "meta": APP_DIR/"03B_FontCLIP_run_metadata_Aonly.csv",
}
TEXT_FILES = {
    "MiniLM":{
        "trait":APP_DIR/"02B_MiniLM_trait_raw.csv",
        "facet":APP_DIR/"02B_MiniLM_facet_raw.csv",
        "dim":APP_DIR/"02B_MiniLM_5D_raw.csv",
    },
    "mDeBERTa":{
        "trait":APP_DIR/"02B_mDeBERTa_trait_raw.csv",
        "facet":APP_DIR/"02B_mDeBERTa_facet_raw.csv",
        "dim":APP_DIR/"02B_mDeBERTa_5D_raw.csv",
    },
}

st.set_page_config(page_title="03C CLIP Wordmark Analyzer", layout="wide")
st.title("03C · CLIP Wordmark Analyzer")
st.caption("FontCLIP과 동일한 흑백 영문 로고타입을 범용 CLIP ViT-B/32로 평가하는 사전고정 비교실험")

st.info(
    "이 실험의 목적은 '일치가 나오는 모델을 찾기'가 아니라, "
    "**동일 이미지·동일 Aaker 42 traits·동일 `{trait} font` prompt**에서 "
    "FontCLIP과 범용 CLIP의 결과가 어떻게 달라지는지 확인하는 것입니다."
)

@st.cache_data
def load_mapping():
    df=pd.read_csv(MAPPING_FILE)
    df["Trait_Key"]=(df["Trait"].str.lower()
                       .str.replace("-","_",regex=False)
                       .str.replace(" ","_",regex=False))
    df["Facet_Key"]=(df["Facet"].str.lower()
                       .str.replace("-","_",regex=False)
                       .str.replace(" ","_",regex=False))
    return df.sort_values("Item_No").reset_index(drop=True)

@st.cache_data
def load_review():
    return pd.read_csv(REVIEW_FILE)

@st.cache_data
def load_fontclip():
    return {
        "review":pd.read_csv(FONT_FILES["review"]),
        "trait":pd.read_csv(FONT_FILES["trait"]),
        "facet":pd.read_csv(FONT_FILES["facet"]),
        "dim":pd.read_csv(FONT_FILES["dim"]),
        "meta":pd.read_csv(FONT_FILES["meta"]),
    }

@st.cache_data
def load_text(model_label):
    F=TEXT_FILES[model_label]
    return {
        "trait":pd.read_csv(F["trait"]),
        "facet":pd.read_csv(F["facet"]),
        "dim":pd.read_csv(F["dim"]),
    }

@st.cache_resource(show_spinner=False)
def load_clip():
    processor=CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model=CLIPModel.from_pretrained(CLIP_MODEL_ID)
    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")
    model.to(device).eval()
    return processor,model,device

def norm_brand(x):
    x=str(x).strip().casefold()
    return re.sub(r"[^0-9a-z가-힣]+","",x)

def l2(x,axis=-1,eps=1e-12):
    n=np.linalg.norm(x,axis=axis,keepdims=True)
    return x/np.maximum(n,eps)

def image_from_zip(zf, name):
    with zf.open(name) as f:
        im=Image.open(io.BytesIO(f.read())).convert("RGB")
    return im

def get_font_path():
    candidates=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

def render_reference_wordmark(text, size=1024, maxw=800):
    im=Image.new("RGB",(size,size),"white")
    draw=ImageDraw.Draw(im)
    fp=get_font_path()
    if fp is None:
        return None
    # Find largest size fitting max width and a conservative max height.
    fs=220
    while fs>=18:
        font=ImageFont.truetype(fp,fs)
        box=draw.textbbox((0,0),text,font=font)
        w,h=box[2]-box[0],box[3]-box[1]
        if w<=maxw and h<=320:
            break
        fs-=4
    font=ImageFont.truetype(fp,fs)
    box=draw.textbbox((0,0),text,font=font)
    w,h=box[2]-box[0],box[3]-box[1]
    x=(size-w)/2-box[0]
    y=(size-h)/2-box[1]
    draw.text((x,y),text,fill="black",font=font)
    return im

def encode_images(processor,model,device,images,batch=16):
    feats=[]
    for s in range(0,len(images),batch):
        x=processor(images=images[s:s+batch],return_tensors="pt")
        pix=x["pixel_values"].to(device)
        with torch.no_grad():
            f=model.get_image_features(pixel_values=pix)
        f=f.detach().cpu().numpy()
        feats.append(l2(f))
    return np.vstack(feats)

def encode_texts(processor,model,device,texts,batch=64):
    feats=[]
    for s in range(0,len(texts),batch):
        x=processor(text=texts[s:s+batch],return_tensors="pt",padding=True,truncation=True)
        ids=x["input_ids"].to(device)
        mask=x["attention_mask"].to(device)
        with torch.no_grad():
            f=model.get_text_features(input_ids=ids,attention_mask=mask)
        f=f.detach().cpu().numpy()
        feats.append(l2(f))
    return np.vstack(feats)

def build_profiles(review,mapping,sim):
    # sim: n_logo x 42
    trait_rows=[]
    for i,r in review.reset_index(drop=True).iterrows():
        row={
            "Filename":r["Filename"],"Brand":r["Brand"],"Eligibility":r["Eligibility"],
            "Neutral_Text":r.get("Neutral_Text",""),"Researcher_Note":r.get("Researcher_Note",""),
        }
        for j,a in mapping.iterrows():
            row["Raw_"+a["Trait_Key"]]=float(sim[i,j])
        trait_rows.append(row)
    trait=pd.DataFrame(trait_rows)

    facet_rows=[]
    for _,r in trait.iterrows():
        out={c:r[c] for c in ["Filename","Brand","Eligibility","Neutral_Text","Researcher_Note"]}
        for (dim,facet,fkey),g in mapping.groupby(["Dimension","Facet","Facet_Key"],sort=False):
            cols=["Raw_"+x for x in g["Trait_Key"]]
            out["Raw_"+fkey]=float(pd.to_numeric(r[cols]).mean())
        facet_rows.append(out)
    facet=pd.DataFrame(facet_rows)

    dim_rows=[]
    for _,r in facet.iterrows():
        out={c:r[c] for c in ["Filename","Brand","Eligibility","Neutral_Text","Researcher_Note"]}
        for d in DIMENSIONS:
            fkeys=(mapping.loc[mapping.Dimension==d,["Facet_Key"]].drop_duplicates()["Facet_Key"].tolist())
            out[d]=float(pd.to_numeric(r[["Raw_"+x for x in fkeys]]).mean())
        dim_rows.append(out)
    dim=pd.DataFrame(dim_rows)
    return trait,facet,dim

def reference_center(df,coord_cols,eligibility_rule):
    out=df.copy()
    if eligibility_rule=="A only":
        ref=out[out.Eligibility=="A"]
    else:
        ref=out[out.Eligibility.isin(["A","B"])]
    means=ref[coord_cols].mean(axis=0)
    for c in coord_cols:
        out[c.replace("Raw_","Centered_") if c.startswith("Raw_") else "Centered_"+c]=out[c]-means[c]
    return out,means

def profile_corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.std(a)==0 or np.std(b)==0: return np.nan
    return float(np.corrcoef(a,b)[0,1])

def profile_cos(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den else np.nan

def align_and_center(left,right,left_cols,right_cols,left_brand="Brand",right_brand="Brand"):
    L=left.copy(); R=right.copy()
    L["_key"]=L[left_brand].map(norm_brand); R["_key"]=R[right_brand].map(norm_brand)
    common=sorted(set(L["_key"]) & set(R["_key"]))
    L=L[L._key.isin(common)].set_index("_key").loc[common].reset_index()
    R=R[R._key.isin(common)].set_index("_key").loc[common].reset_index()
    A=L[left_cols].astype(float).copy()
    B=R[right_cols].astype(float).copy()
    A=A-A.mean(axis=0)
    B=B-B.mean(axis=0)
    return common,L,R,A,B

def pair_profile_table(common,L,R,A,B,level):
    rows=[]
    for i,key in enumerate(common):
        a=A.iloc[i].to_numpy(float); b=B.iloc[i].to_numpy(float)
        rows.append({
            "Brand_Left":L.iloc[i]["Brand"],
            "Brand_Right":R.iloc[i]["Brand"],
            "Level":level,
            "Profile_Pearson_r":profile_corr(a,b),
            "Profile_Cosine":profile_cos(a,b),
            "Direction_Agreement":int(np.sum(np.sign(a)==np.sign(b))),
            "N_Coordinates":len(a),
        })
    return pd.DataFrame(rows)

def permutation_test(A,B,n_perm=10000,seed=20260927):
    A=A.to_numpy(float); B=B.to_numpy(float); n=len(A)
    M=np.array([[profile_corr(A[i],B[j]) for j in range(n)] for i in range(n)])
    obs=float(np.nanmean(np.diag(M)))
    off=float(np.nanmean(M[~np.eye(n,dtype=bool)]))
    rng=np.random.default_rng(seed)
    vals=np.empty(n_perm)
    for k in range(n_perm):
        p=rng.permutation(n)
        vals[k]=np.nanmean([M[i,p[i]] for i in range(n)])
    p=(np.sum(vals>=obs)+1)/(n_perm+1)
    return {"Matched_Mean_r":obs,"Nonmatching_Mean_r":off,"Permutation_p_one_sided":p,"N":n,"N_Permutations":n_perm}

def compare_hierarchy(left,right,mapping,left_prefix="",right_prefix="",n_perm=10000,label=""):
    trait_keys=mapping["Trait_Key"].tolist()
    facet_keys=mapping[["Dimension","Facet","Facet_Key"]].drop_duplicates()["Facet_Key"].tolist()
    specs=[
        ("42-trait",[left_prefix+x for x in trait_keys],[right_prefix+x for x in trait_keys]),
        ("15-facet",[left_prefix+x for x in facet_keys],[right_prefix+x for x in facet_keys]),
        ("5D",DIMENSIONS,DIMENSIONS),
    ]
    out={}
    for level,lc,rc in specs:
        common,L,R,A,B=align_and_center(left[level],right[level],lc,rc)
        out[level]=pair_profile_table(common,L,R,A,B,level)
        out[level+"_perm"]=pd.DataFrame([permutation_test(A,B,n_perm)])
    return out

def load_bundled_fontclip(mapping):
    return {
        "42-trait":pd.read_csv(FONT_FILES["trait"]),
        "15-facet":pd.read_csv(FONT_FILES["facet"]),
        "5D":pd.read_csv(FONT_FILES["dim"]),
    }

def load_bundled_text(label):
    F=TEXT_FILES[label]
    return {
        "42-trait":pd.read_csv(F["trait"]),
        "15-facet":pd.read_csv(F["facet"]),
        "5D":pd.read_csv(F["dim"]),
    }

def clip_hierarchy_dict(trait,facet,dim,elig):
    if elig=="A only":
        filt=lambda df: df[df.Eligibility=="A"].copy()
    else:
        filt=lambda df: df[df.Eligibility.isin(["A","B"])].copy()
    return {"42-trait":filt(trait),"15-facet":filt(facet),"5D":filt(dim)}

def text_hierarchy_dict(text):
    return text

def zip_csvs(files:Dict[str,pd.DataFrame],meta:dict):
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,"w",zipfile.ZIP_DEFLATED) as zf:
        for name,df in files.items():
            zf.writestr(name,df.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("03C_99_run_metadata.json",json.dumps(meta,ensure_ascii=False,indent=2))
    mem.seek(0)
    return mem.getvalue()

mapping=load_mapping()
review_fixed=load_review()

st.markdown("### 사전고정 비교조건")
st.markdown(
    """
- **CLIP:** `openai/clip-vit-base-patch32`
- **이미지:** 03B와 동일한 흰 배경·검정 영문 로고타입 1024×1024
- **Aaker:** 42 traits → 15 facets → 5 dimensions
- **Prompt:** `{trait} font`
- **Negative prompt 없음**
- **Reference:** A-only 본분석 / A+B 민감도
- **Text:** 사전 실행한 NLI MiniLM + mDeBERTa 둘 다 비교
- 결과 확인 후 prompt/model을 바꾸지 않음
    """
)

logo_zip=st.file_uploader("`logo_dataset_68_fixed.zip` 업로드",type=["zip"])
ref_rule=st.radio("Reference pool",["A only","A+B"],index=0,horizontal=True)
nperm=st.selectbox("Permutation 횟수",[1000,5000,10000,50000],index=2)
do_ref_font=st.checkbox("DejaVu Sans reference-font sensitivity",value=True)

if st.button("03C CLIP-wordmark 분석 실행",type="primary",disabled=logo_zip is None):
    try:
        processor,model,device=load_clip()
        with zipfile.ZipFile(io.BytesIO(logo_zip.getvalue())) as zf:
            # Use exact fixed review order and names.
            imgs=[]
            valid_rows=[]
            for _,r in review_fixed.iterrows():
                if r["Filename"] not in zf.namelist():
                    st.warning(f"ZIP에서 누락: {r['Filename']}")
                    continue
                imgs.append(image_from_zip(zf,r["Filename"]))
                valid_rows.append(r.to_dict())
        review=pd.DataFrame(valid_rows).reset_index(drop=True)

        prompts=[f"{t} font" for t in mapping["Trait"]]
        with st.spinner(f"CLIP {CLIP_MODEL_ID} image/text embedding 계산"):
            I=encode_images(processor,model,device,imgs)
            T=encode_texts(processor,model,device,prompts)
        sim=I@T.T
        trait,facet,dim=build_profiles(review,mapping,sim)

        trait_cols=["Raw_"+x for x in mapping["Trait_Key"]]
        facet_keys=mapping[["Dimension","Facet","Facet_Key"]].drop_duplicates()["Facet_Key"].tolist()
        facet_cols=["Raw_"+x for x in facet_keys]
        trait_cent,trait_means=reference_center(trait,trait_cols,ref_rule)
        facet_cent,facet_means=reference_center(facet,facet_cols,ref_rule)
        dim_cent,dim_means=reference_center(dim,DIMENSIONS,ref_rule)

        # CLIP vs FontCLIP, same eligible set.
        clipH=clip_hierarchy_dict(trait,facet,dim,ref_rule)
        fontH=load_bundled_fontclip(mapping)
        if ref_rule=="A only":
            for k in fontH:
                if "Eligibility" in fontH[k].columns:
                    fontH[k]=fontH[k][fontH[k].Eligibility=="A"].copy()
        else:
            for k in fontH:
                if "Eligibility" in fontH[k].columns:
                    fontH[k]=fontH[k][fontH[k].Eligibility.isin(["A","B"])].copy()

        # Custom comparison because prefixes differ.
        trait_keys=mapping["Trait_Key"].tolist()
        facet_keys=mapping[["Dimension","Facet","Facet_Key"]].drop_duplicates()["Facet_Key"].tolist()
        model_cmp={}
        for level,lc,rc in [
            ("42-trait",["Raw_"+x for x in trait_keys],["Raw_"+x for x in trait_keys]),
            ("15-facet",["Raw_"+x for x in facet_keys],["Raw_"+x for x in facet_keys]),
            ("5D",DIMENSIONS,DIMENSIONS),
        ]:
            common,L,R,A,B=align_and_center(clipH[level],fontH[level],lc,rc)
            model_cmp[level]=pair_profile_table(common,L,R,A,B,level)
            model_cmp[level+"_perm"]=pd.DataFrame([permutation_test(A,B,nperm)])

        # Text vs CLIP for both pre-registered NLI models.
        text_results={}
        for label in ["MiniLM","mDeBERTa"]:
            txt=load_bundled_text(label)
            tmp={}
            for level,lc,rc in [
                ("42-trait",trait_keys,["Raw_"+x for x in trait_keys]),
                ("15-facet",facet_keys,["Raw_"+x for x in facet_keys]),
                ("5D",DIMENSIONS,DIMENSIONS),
            ]:
                common,L,R,A,B=align_and_center(txt[level],clipH[level],lc,rc)
                tmp[level]=pair_profile_table(common,L,R,A,B,level)
                tmp[level+"_perm"]=pd.DataFrame([permutation_test(A,B,nperm)])
            text_results[label]=tmp

        # Reference-font sensitivity
        ref_sens=pd.DataFrame()
        if do_ref_font:
            ref_imgs=[]
            ref_rows=[]
            for _,r in review.iterrows():
                rim=render_reference_wordmark(str(r.get("Neutral_Text") or r["Brand"]))
                if rim is not None:
                    ref_imgs.append(rim); ref_rows.append(r)
            if ref_imgs:
                RI=encode_images(processor,model,device,ref_imgs)
                rsim=RI@T.T
                rreview=pd.DataFrame([x.to_dict() for x in ref_rows]).reset_index(drop=True)
                _,_,rdim=build_profiles(rreview,mapping,rsim)
                actual=dim.set_index("Brand")[DIMENSIONS]
                refd=rdim.set_index("Brand")[DIMENSIONS]
                cc=actual.index.intersection(refd.index)
                rows=[]
                for b in cc:
                    rows.append({
                        "Brand":b,
                        "Eligibility":dim.loc[dim.Brand==b,"Eligibility"].iloc[0],
                        "Actual_vs_ReferenceFont_5D_Cosine":profile_cos(actual.loc[b],refd.loc[b]),
                        "Actual_vs_ReferenceFont_5D_Pearson":profile_corr(actual.loc[b],refd.loc[b]),
                    })
                ref_sens=pd.DataFrame(rows)

        st.session_state["03C"]={
            "review":review,"trait":trait,"trait_cent":trait_cent,
            "facet":facet,"facet_cent":facet_cent,
            "dim":dim,"dim_cent":dim_cent,
            "model_cmp":model_cmp,"text_results":text_results,
            "ref_sens":ref_sens,"device":str(device),"ref_rule":ref_rule,
            "nperm":nperm,
        }
        st.success("03C CLIP-wordmark 분석 완료")
    except Exception as e:
        st.exception(e)

if "03C" in st.session_state:
    R=st.session_state["03C"]
    st.header("1. CLIP 내부 진단")
    dd=R["dim"][DIMENSIONS]
    corr=dd.corr()
    mean_corr=float(corr.to_numpy()[np.triu_indices(5,1)].mean())
    st.metric("CLIP 5D 차원 간 평균상관",f"{mean_corr:.3f}")
    prim=dd.idxmax(axis=1).value_counts().rename_axis("Primary_Dimension").reset_index(name="N")
    st.dataframe(prim,use_container_width=True,hide_index=True)

    st.header("2. CLIP ↔ FontCLIP 동일 wordmark 비교")
    rows=[]
    for level in ["42-trait","15-facet","5D"]:
        p=R["model_cmp"][level+"_perm"].iloc[0]
        rows.append({"Level":level,**p.to_dict()})
    model_sum=pd.DataFrame(rows)
    st.dataframe(model_sum.style.format({
        "Matched_Mean_r":"{:.3f}","Nonmatching_Mean_r":"{:.3f}","Permutation_p_one_sided":"{:.4f}"
    }),use_container_width=True,hide_index=True)

    st.header("3. Text NLI ↔ CLIP 계산 일치도")
    all_summaries=[]
    for label in ["MiniLM","mDeBERTa"]:
        for level in ["42-trait","15-facet","5D"]:
            p=R["text_results"][label][level+"_perm"].iloc[0]
            all_summaries.append({"Text_Model":label,"Level":level,**p.to_dict()})
    text_sum=pd.DataFrame(all_summaries)
    st.dataframe(text_sum.style.format({
        "Matched_Mean_r":"{:.3f}","Nonmatching_Mean_r":"{:.3f}","Permutation_p_one_sided":"{:.4f}"
    }),use_container_width=True,hide_index=True)

    if not R["ref_sens"].empty:
        st.header("4. Reference-font sensitivity")
        st.dataframe(R["ref_sens"].sort_values("Actual_vs_ReferenceFont_5D_Cosine"),
                     use_container_width=True,hide_index=True)

    files={
        "03C_01_logo_review.csv":R["review"],
        "03C_02_trait_scores_raw.csv":R["trait"],
        "03C_03_trait_scores_reference_centered.csv":R["trait_cent"],
        "03C_04_facet_profiles_raw.csv":R["facet"],
        "03C_05_facet_profiles_reference_centered.csv":R["facet_cent"],
        "03C_06_5D_profiles_raw.csv":R["dim"],
        "03C_07_5D_profiles_reference_centered.csv":R["dim_cent"],
        "03C_08_CLIP_vs_FontCLIP_42trait.csv":R["model_cmp"]["42-trait"],
        "03C_09_CLIP_vs_FontCLIP_15facet.csv":R["model_cmp"]["15-facet"],
        "03C_10_CLIP_vs_FontCLIP_5D.csv":R["model_cmp"]["5D"],
        "03C_11_CLIP_vs_FontCLIP_summary.csv":model_sum,
        "03C_12_Text_vs_CLIP_summary.csv":text_sum,
    }
    for label in ["MiniLM","mDeBERTa"]:
        for level,key in [("42trait","42-trait"),("15facet","15-facet"),("5D","5D")]:
            files[f"03C_Text_{label}_vs_CLIP_{level}.csv"]=R["text_results"][label][key]
    if not R["ref_sens"].empty:
        files["03C_13_reference_font_sensitivity.csv"]=R["ref_sens"]

    meta={
        "App_Version":APP_VERSION,
        "CLIP_Model":CLIP_MODEL_ID,
        "Image_Input":"same standardized B/W wordmarks as 03B FontCLIP",
        "Prompt":"{trait} font",
        "Negative_Prompt":False,
        "Aaker_Structure":"42 traits -> 15 facets -> 5 dimensions",
        "Reference_Rule":R["ref_rule"],
        "Permutation_N":R["nperm"],
        "Device":R["device"],
        "Text_Models":["MiniLM","mDeBERTa"],
        "Purpose":"pre-registered model comparison, not outcome-seeking model selection",
    }
    data=zip_csvs(files,meta)
    st.download_button(
        "03C 전체 결과 ZIP 다운로드",
        data=data,
        file_name="03C_clip_wordmark_results_v1_0.zip",
        mime="application/zip"
    )
