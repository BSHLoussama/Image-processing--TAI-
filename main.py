import streamlit as st
import pandas as pd
import numpy as np
import math
import tempfile
import cv2
import subprocess
import os
import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------------------
# Shared Utility Functions
# -------------------------------------------
def custom_min(data): return min(data)
def custom_max(data): return max(data)
def custom_mean(data): return sum(data)/len(data)
def custom_median(data):
    sd = sorted(data); n = len(sd); m = n//2
    return sd[m] if n%2 else (sd[m-1]+sd[m])/2
def custom_quantile(data,q):
    sd = sorted(data); pos = (len(sd)-1)*q
    lo,hi = math.floor(pos), math.ceil(pos)
    if lo==hi: return sd[int(pos)]
    frac = pos-lo
    return sd[lo]*(1-frac) + sd[hi]*frac
def custom_std(data):
    mu = custom_mean(data)
    return math.sqrt(sum((x-mu)**2 for x in data)/len(data))
def custom_mode(data):
    freq = {}
    for x in data: freq[x] = freq.get(x,0)+1
    return max(freq, key=freq.get)

@st.cache_data
def load_image(f):
    img = Image.open(f).convert("RGB")
    w,h = img.size; mw,mh = 800,800
    if w>mw or h>mh:
        sc = min(mw/w, mh/h)
        img = img.resize((int(w*sc), int(h*sc)),
                         getattr(Image,'Resampling',Image).LANCZOS)
    return img

@st.cache_data
def convert_to_grayscale(img):
    return img.convert("L")

@st.cache_data
def translate_brightness(img,value):
    arr = np.clip(np.array(img).astype(int)+value,0,255).astype(np.uint8)
    return Image.fromarray(arr).convert(img.mode)

@st.cache_data
def equalize_histogram(img):
    eq = ImageOps.equalize(img)
    return eq.convert(img.mode)


# -------------------------------------------
# Compute Region Properties & Freeman Chain Codes
# -------------------------------------------
def compute_props_and_chain(bin_img: np.ndarray):
    n_labels, labels, stats, cents = cv2.connectedComponentsWithStats(
        (bin_img>0).astype(np.uint8), connectivity=8)
    props, chains = [], []
    dirs = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
    arrows_map = {i:a for i,a in enumerate('→↗↑↖←↙↓↘')}
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = cents[i]
        mask_i = ((labels==i).astype(np.uint8))*255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if cnts:
            cnt = cnts[0]
            peri = round(cv2.arcLength(cnt, True), 2)
        else:
            cnt = []; peri = 0.0
        props.append({
            "Region": i,
            "Area": area,
            "Perimeter": peri,
            "CentroidX": round(cx,2),
            "CentroidY": round(cy,2)
        })
        code, arrow = [], []
        for j in range(len(cnt)-1):
            x1,y1 = cnt[j][0]; x2,y2 = cnt[j+1][0]
            dr, dc = (y2-y1, x2-x1)
            best = min(range(8),
                       key=lambda k: (dr-dirs[k][0])**2 + (dc-dirs[k][1])**2)
            code.append(str(best)); arrow.append(arrows_map[best])
        chains.append({
            "Region": i,
            "ChainCode": "".join(code),
            "ArrowCode": "".join(arrow)
        })
    return pd.DataFrame(props), pd.DataFrame(chains)


# -------------------------------------------
# TP1: Histogram Operations
# -------------------------------------------
def compute_histogram(img):
    gray = np.array(img.convert("L"))
    return np.histogram(gray.flatten(), bins=256, range=(0,256))

def plot_histogram(img,title,color_space='RGB',show_legend=True):
    arr = np.array(img)
    plt.figure(figsize=(8,3))
    if arr.ndim==2:
        plt.hist(arr.flatten(), bins=128, color='gray', alpha=0.7)
    else:
        cols=['r','g','b']
        names = (['Red','Green','Blue'] if color_space=='RGB'
                 else ['Hue','Sat','Val'] if color_space=='HSV'
                 else ['L','A','B'])
        for i,c in enumerate(cols):
            plt.hist(arr[:,:,i].flatten(), bins=128,
                     color=c, alpha=0.5, label=names[i])
        if show_legend: plt.legend()
    plt.title(f"Histogram - {title}")
    plt.xlabel("Intensity"); plt.ylabel("Frequency")
    st.pyplot(plt); plt.clf()

def plot_normalized_histogram(img,title="Normalized Histogram"):
    hist,bins = compute_histogram(img)
    norm = hist/(img.size[0]*img.size[1])
    plt.figure(figsize=(8,3))
    plt.bar(bins[:-1], norm, width=1.0, color='gray', alpha=0.7)
    plt.title(title); plt.xlabel("Gray Level"); plt.ylabel("Probability")
    st.pyplot(plt); plt.clf()

def plot_cumulative_histogram(img,title="Cumulative Histogram"):
    hist,bins = compute_histogram(img)
    cum = np.cumsum(hist)
    plt.figure(figsize=(8,3))
    plt.plot(bins[:-1], cum, color='blue')
    plt.title(title); plt.xlabel("Gray Level"); plt.ylabel("Cumulative Freq")
    st.pyplot(plt); plt.clf()

def plot_normalized_cumulative_histogram(img,title="Normalized Cumulative Histogram"):
    hist,bins = compute_histogram(img)
    cum = np.cumsum(hist); norm_cum = cum/cum[-1]
    plt.figure(figsize=(8,3))
    plt.plot(bins[:-1], norm_cum, color='green')
    plt.title(title); plt.xlabel("Gray Level"); plt.ylabel("Cumulative Prob")
    st.pyplot(plt); plt.clf()

def process_histogram_section(img):
    st.header("TP1: Histogram Operations")
    gray = convert_to_grayscale(img)
    st.image(gray, caption="Grayscale", use_container_width=True)
    plot_histogram(gray, "Original", color_space='GRAY', show_legend=False)
    plot_normalized_histogram(gray)
    plot_cumulative_histogram(gray)
    plot_normalized_cumulative_histogram(gray)


# -------------------------------------------
# TP2: Image Transformations
# -------------------------------------------
def histogram_inversion(img):
    inv = 255 - np.array(img)
    return Image.fromarray(inv.astype(np.uint8)).convert(img.mode)

def dynamic_range_expansion(img):
    arr = np.array(img).astype(float)
    mn,mx = arr.min(), arr.max()
    if mx==mn: return img
    stretched = (arr-mn)*255.0/(mx-mn)
    return Image.fromarray(np.clip(stretched,0,255).astype(np.uint8)).convert(img.mode)

@st.cache_data
def equalize_histogram_manual(img):
    gray = np.array(img.convert("L"))
    hist,bins = np.histogram(gray.flatten(),256,[0,256])
    cdf = hist.cumsum(); cdf_norm = cdf*255/cdf[-1]
    eq = np.interp(gray.flatten(), bins[:-1], cdf_norm).reshape(gray.shape)
    return Image.fromarray(np.clip(eq,0,255).astype(np.uint8)).convert(img.mode)

def histogram_specification(src,ref):
    s = np.array(src.convert("L"))
    r = np.array(ref.convert("L"))
    sh,_ = np.histogram(s.flatten(),256,[0,256])
    rh,_ = np.histogram(r.flatten(),256,[0,256])
    scdf = sh.cumsum()/sh.sum(); rcdf = rh.cumsum()/rh.sum()
    mapping = np.zeros(256,dtype=np.uint8)
    for i in range(256):
        mapping[i] = np.argmin(np.abs(rcdf-scdf[i]))
    spec = mapping[s]
    return Image.fromarray(spec).convert(src.mode)

def process_transforms_section(img):
    st.header("TP2: Image Transformations")
    gray = convert_to_grayscale(img)
    st.image(gray, caption="Grayscale", use_container_width=True)
    op = st.radio("Operation", [
        "Translation","Inversion","Dynamic Range Expansion",
        "Hist. Equalization","Hist. Specification"
    ])
    res = None
    if op=="Translation":
        v = st.slider("Brightness Δ", -100,100,0)
        res = translate_brightness(gray,v)
    elif op=="Inversion":
        res = histogram_inversion(gray)
    elif op=="Dynamic Range Expansion":
        res = dynamic_range_expansion(gray)
    elif op=="Hist. Equalization (Manual)":
        res = equalize_histogram_manual(gray)
    else:
        ref = st.file_uploader(
            "Reference Image",
            type=["png","jpg","jpeg","bmp","tif","tiff"]
        )
        if ref:
            res = histogram_specification(gray, load_image(ref))
        else:
            st.error("Upload reference image.")
    if res:
        st.image(res, caption=f"Result: {op}", use_container_width=True)
        plot_histogram(res, op, color_space='GRAY', show_legend=False)
        plot_normalized_histogram(res)
        plot_cumulative_histogram(res)
        plot_normalized_cumulative_histogram(res)


# -------------------------------------------
# TP3: Color Processing (OpenCV HSV/LAB)
# -------------------------------------------
@st.cache_data
def convert_to_hsv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

@st.cache_data
def convert_to_lab(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)

@st.cache_data
def uniform_quantization(img,levels=4):
    arr = np.array(img)
    q = (arr//(256//levels))*(256//levels)
    return Image.fromarray(q.astype(np.uint8)).convert(img.mode)

@st.cache_data
def kmeans_quantization(img,clusters=4):
    arr = np.array(img); data = arr.reshape(-1,3)
    sample = data if len(data)<=100_000 else data[
        np.random.choice(len(data),100_000, replace=False)]
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init='auto')
    kmeans.fit(sample); labels = kmeans.predict(data)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    quant = centers[labels].reshape(arr.shape)
    return Image.fromarray(quant).convert(img.mode)

@st.cache_data
def median_cut_quantization(img,clusters=4):
    return kmeans_quantization(img,clusters)

def apply_median_filter(img, ksize):
    arr = np.array(img)
    if ksize % 2 == 0: ksize += 1
    if arr.ndim == 2:
        blurred = median_blur(arr, ksize)
    else:
        chans = [median_blur(arr[:,:,c], ksize) for c in range(arr.shape[2])]
        blurred = np.stack(chans, axis=2)
    return Image.fromarray(blurred.astype(np.uint8)).convert(img.mode)

def process_color_section(img):
    st.header("TP3: Color Processing")
    st.image(img, caption="Original", use_container_width=True)
    plot_histogram(img, "Original Color")
    if st.button("Equalize Color"):
        img = equalize_histogram(img)
        st.image(img, caption="Equalized", use_container_width=True)
    plot_histogram(img, "Equalized Color")
    m = st.selectbox("Quant Method", ["Uniform","K-Means","Median Cut"])
    lv = st.slider("Levels", 2,16,4)
    if st.button("Apply Quant"):
        qt = {
            "Uniform": uniform_quantization,
            "K-Means": kmeans_quantization,
            "Median Cut": median_cut_quantization
        }[m](img, lv)
        st.image(qt, caption=f"{m} Quantized", use_container_width=True)
        plot_histogram(qt, f"{m} Quantized")
    st.markdown("### Color Space Channels")
    spaces = {
        "RGB": img,
        "HSV": Image.fromarray(convert_to_hsv(img)),
        "LAB": Image.fromarray(convert_to_lab(img))
    }
    names = {"RGB":["Red","Green","Blue"],
             "HSV":["Hue","Sat","Val"],
             "LAB":["L","A","B"]}
    for sp,sp_img in spaces.items():
        st.subheader(sp)
        chs = sp_img.split()
        cols = st.columns(3)
        for i,ch in enumerate(chs):
            with cols[i]:
                st.image(ch, caption=names[sp][i], use_container_width=True)
                plot_histogram(ch, f"{sp}-{names[sp][i]}")


# -------------------------------------------
# TP4: Convolution / Filtering & Custom Operators
# -------------------------------------------
def convolve2d_scratch(img_arr,kernel,boundary='symm'):
    K = np.flipud(np.fliplr(kernel))
    kH,kW = K.shape; iH,iW = img_arr.shape
    ph,pw = (kH-1)//2,(kW-1)//2
    pad = np.pad(img_arr, ((ph,ph),(pw,pw)),
                 mode='reflect' if boundary=='symm' else 'constant')
    out = np.zeros_like(img_arr, dtype=float)
    for i in range(iH):
        for j in range(iW):
            out[i,j] = np.sum(pad[i:i+kH, j:j+kW] * K)
    return out

def median_blur(arr, ksize):
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize//2
    padded = np.pad(arr, pad, mode='reflect')
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i,j] = np.median(padded[i:i+ksize, j:j+ksize])
    return out

def sobel_filters(gray):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=float)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=float)
    gx = convolve2d_scratch(gray.astype(float), Kx)
    gy = convolve2d_scratch(gray.astype(float), Ky)
    return gx, gy

def gradient_magnitude(gx, gy):
    mag = np.hypot(gx, gy)
    mag = mag/(mag.max() or 1)*255
    return mag.astype(np.uint8)

def threshold_binary(arr, t_low, t_max=255):
    return ((arr > t_low)*t_max).astype(np.uint8)

def canny_edge(gray, t_low, t_high):
    g = np.array([[2,4,5,4,2],[4,9,12,9,4],
                  [5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]],dtype=float)
    g /= g.sum()
    smooth = convolve2d_scratch(gray.astype(float), g)
    gx, gy = sobel_filters(smooth)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy,gx)*(180/np.pi)
    ang[ang<0] += 180
    M,N = gray.shape
    Z = np.zeros((M,N),dtype=np.uint8)
    for i in range(1,M-1):
        for j in range(1,N-1):
            a = ang[i,j]
            q=r=0
            if (0<=a<22.5) or (157.5<=a<=180):
                q,r = mag[i,j+1], mag[i,j-1]
            elif 22.5<=a<67.5:
                q,r = mag[i+1,j-1], mag[i-1,j+1]
            elif 67.5<=a<112.5:
                q,r = mag[i+1,j],   mag[i-1,j]
            else:
                q,r = mag[i-1,j-1], mag[i+1,j+1]
            Z[i,j] = mag[i,j] if (mag[i,j]>=q and mag[i,j]>=r) else 0
    strong, weak = 255, 75
    res = np.zeros((M,N),dtype=np.uint8)
    res[Z>=t_high] = strong
    res[(Z>=t_low)&(Z<t_high)] = weak
    for i in range(1,M-1):
        for j in range(1,N-1):
            if res[i,j]==weak:
                if np.any(res[i-1:i+2,j-1:j+2]==strong):
                    res[i,j]=strong
                else:
                    res[i,j]=0
    return res

def apply_convolution_scratch(img,kernel):
    arr = np.array(img).astype(float)
    if arr.ndim==2:
        conv = convolve2d_scratch(arr,kernel)
        return Image.fromarray(np.clip(conv,0,255).astype(np.uint8)).convert(img.mode)
    chans = [np.clip(convolve2d_scratch(arr[:,:,c],kernel),0,255).astype(np.uint8)
             for c in range(3)]
    return Image.fromarray(np.dstack(chans))

def process_convolution_section(img):
    st.header("TP4: Convolution / Filtering")
    op = st.radio("Type", ["Convolution","Median"])
    res = None
    if op=="Convolution":
        kc = st.selectbox("Kernel", ["Mean(3x3)","Gauss(3x3)","Laplacian","Custom"])
        if kc=="Mean(3x3)":
            K = np.ones((3,3))/9
        elif kc=="Gauss(3x3)":
            K = np.array([[1,2,1],[2,4,2],[1,2,1]],float)/16
        elif kc=="Laplacian":
            K = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],float)
        else:
            inp = st.text_area("9 comma-separated vals",
                               "0,-1,0,-1,5,-1,0,-1,0")
            try:
                vals = [float(x) for x in inp.split(",")]
                K = np.array(vals).reshape(3,3)
            except:
                st.error("Invalid kernel"); K=None
        if 'K' in locals() and K is not None:
            res = apply_convolution_scratch(img,K)
    else:
        s = st.slider("Size",3,11,3,step=2)
        res = apply_median_filter(img,s)

    if res:
        st.image(res, caption="Filtered", use_container_width=True)
        plot_histogram(res, "Filtered",
                       color_space='GRAY' if res.mode=='L' else 'RGB')


# -------------------------------------------
# TP5: Contours (uses our custom Canny/Sobel)
# -------------------------------------------
def detect_contours_opencv(img, method="Canny", t_low=50, t_high=150, min_area=0):
    gray = np.array(img.convert("L"))
    if method=="Canny":
        edges = canny_edge(gray, t_low, t_high)
    else:
        gx, gy = sobel_filters(gray)
        mag     = gradient_magnitude(gx, gy)
        edges   = threshold_binary(mag, t_low, 255)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    filtered = [c for c in cnts if cv2.contourArea(c)>=min_area]
    canvas = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()
    cv2.drawContours(overlay, filtered, -1, (0,255,0), 1)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    stats = []
    for idx, c in enumerate(filtered):
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        m = cv2.moments(c)
        cx = (m["m10"]/m["m00"]) if m["m00"] else 0
        cy = (m["m01"]/m["m00"]) if m["m00"] else 0
        stats.append({
            "ID": idx+1,
            "Area": round(area,2),
            "Perimeter": round(peri,2),
            "CentroidX": round(cx,2),
            "CentroidY": round(cy,2)
        })
    return Image.fromarray(overlay), Image.fromarray(edges), stats

def process_contours_section(img):
    st.header("TP5: Contours")
    m = st.selectbox("Method", ["Canny","Sobel"])
    tl = st.slider("Low Thr",0,255,50)
    th = st.slider("High Thr",0,255,150) if m=="Canny" else 0
    ma = st.slider("Min Area",0,10000,100)
    if st.button("Detect Contours"):
        overlay, edge_map, stats = detect_contours_opencv(
            img, m, tl, th, ma)
        st.image(edge_map, caption="Edge Map", use_container_width=True)
        st.image(overlay, caption=f"Overlay ({len(stats)} regions)",
                 use_container_width=True)
        if stats:
            st.markdown("### Contour Statistics")
            st.dataframe(pd.DataFrame(stats))


# -------------------------------------------
# TP6: Segmentation (Thresholding / K-Means)
# -------------------------------------------
@st.cache_data
def classify_kmeans_gray(img, n_clusters):
    gray = np.array(img.convert("L"))
    data = gray.reshape(-1,1).astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(data)
    label_img = labels.reshape(gray.shape).astype(np.uint8)
    if n_clusters > 1:
        label_img = (label_img * (255 // (n_clusters-1))).astype(np.uint8)
    else:
        label_img = label_img * 255
    return Image.fromarray(label_img)

def binarize_image(img, method='Manual', threshold=128):
    gray = np.array(img.convert("L"))
    if method == 'Manual':
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif method == 'Otsu':
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif method == 'Haris':
        t = int(gray.mean()+10)
        mask = (gray>=t).astype(np.uint8)*255
    else:
        mask = np.zeros_like(gray)
    return Image.fromarray(mask)

def process_segmentation_section(img):
    st.header("TP6: Segmentation")
    method = st.radio("Méthode",
                     ["Manual","Otsu","Haris","Classification"])
    if method in ("Manual","Otsu","Haris"):
        if method=="Manual":
            plot_histogram(convert_to_grayscale(img),
                           "Grayscale", color_space='GRAY', show_legend=False)
            th = st.slider("Threshold",0,255,128)
            bin_img = binarize_image(img,'Manual',th)
        elif method=="Otsu":
            bin_img = binarize_image(img,'Otsu')
        else:
            bin_img = binarize_image(img,'Haris')
        st.image(bin_img, caption="Segmentation par seuillage",
                 use_container_width=True)
    else:
        n = st.slider("Nombre de classes",2,8,2)
        seg = classify_kmeans_gray(img,n)
        st.image(seg, caption=f"Segmentation par classification ({n} clusters)",
                 use_container_width=True)


# -------------------------------------------
# TP7: CC Labeling Only
# -------------------------------------------
def process_tp7_section(img):
    st.header("TP7: CC Labeling")
    gray = convert_to_grayscale(img)
    st.image(gray, caption="Grayscale", use_container_width=True)
    th = st.slider("Threshold",0,255,128)
    bin_img = binarize_image(img,'Manual',th)
    st.image(bin_img, caption="Binary Image", use_container_width=True)
    arr = np.array(bin_img)>0
    n_labels, labels = cv2.connectedComponents(arr.astype(np.uint8))
    colors = np.vstack([[0,0,0],
                        np.random.randint(0,255,(n_labels,3))]).astype(np.uint8)
    colored = colors[labels]
    st.image(colored, caption=f"Labeled ({n_labels-1} components)",
             use_container_width=True)


# -------------------------------------------
# TP8: Region Properties & Freeman Codes
# -------------------------------------------
def process_tp8_section(img):
    st.header("TP8: Region Properties & Freeman Codes")
    gray = convert_to_grayscale(img)
    st.image(gray, caption="Grayscale", use_container_width=True)
    th = st.slider("Threshold",0,255,128)
    bin_img = binarize_image(img,'Manual',th)
    st.image(bin_img, caption="Binary Mask", use_container_width=True)
    arr = np.array(bin_img)>0
    df_props, df_chain = compute_props_and_chain(arr.astype(np.uint8))
    st.markdown("### Area, Perimeter & Centroid")
    st.dataframe(df_props)
    st.download_button("Download Region Properties CSV",
                       df_props.to_csv(index=False),
                       file_name="region_properties.csv",
                       mime="text/csv")
    st.markdown("### Freeman Chain Codes")
    st.dataframe(df_chain)
    st.download_button("Download Chain Codes CSV",
                       df_chain.to_csv(index=False),
                       file_name="chain_codes.csv",
                       mime="text/csv")


# -------------------------------------------
# TP9: Video Processing “From Scratch” via ffmpeg
# -------------------------------------------
def process_video_section():
    st.header("TP9: Video Display")
    vfile = st.file_uploader("Upload Video",
                             type=["mp4","avi","mov","mkv"])
    if not vfile:
        return

    # 1) Write to temp
    tmpdir = tempfile.mkdtemp()
    ext = os.path.splitext(vfile.name)[1].lower()
    vpath = os.path.join(tmpdir, f"input{ext}")
    with open(vpath, "wb") as f:
        f.write(vfile.read())

    # 2) Extract frames with ffmpeg
    out_pattern = os.path.join(tmpdir, "frame_%05d.png")
    try:
        subprocess.run(
            ["ffmpeg", "-i", vpath, "-vsync", "0", out_pattern],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
    except FileNotFoundError:
        st.error("FFmpeg not found on PATH. Please install FFmpeg.")
        return
    except subprocess.CalledProcessError:
        st.error("FFmpeg failed to extract frames.")
        return

    # 3) Gather frames
    frames = sorted(glob.glob(os.path.join(tmpdir, "frame_*.png")))
    total = len(frames)
    if total == 0:
        st.error("No frames extracted.")
        return

    # 4) Slider & display
    idx = st.slider("Select Frame", 0, total-1, 0)
    frame_img = Image.open(frames[idx]).convert("RGB")
    st.image(frame_img, caption=f"Frame {idx}", use_container_width=True)


# -------------------------------------------
# Main Streamlit App
# -------------------------------------------
st.set_page_config(page_title="TP Image & Video App",
                   layout="wide")
st.title("Projet TP Traitement Automatique d'Image\n ### **Preparé par:**  Oussama BOUSSAHLA - Souhil Mokeddem")

mode = st.sidebar.radio("Select Section", [
    "TP1 - Histogram","TP2 - Transform","TP3 - Color",
    "TP4 - Convolution","TP5 - Contours","TP6 - Segmentation",
    "TP7 - CC Labeling","TP8 - Region Props & Codes",
    "TP9 - Video"
])

if mode != "TP9 - Video":
    uploaded = st.sidebar.file_uploader(
        "Upload Image",
        type=["png","jpg","jpeg","bmp","tif","tiff"]
    )
    if uploaded:
        img = load_image(uploaded)
        if mode == "TP1 - Histogram":
            process_histogram_section(img)
        elif mode == "TP2 - Transform":
            process_transforms_section(img)
        elif mode == "TP3 - Color":
            process_color_section(img)
        elif mode == "TP4 - Convolution":
            process_convolution_section(img)
        elif mode == "TP5 - Contours":
            process_contours_section(img)
        elif mode == "TP6 - Segmentation":
            process_segmentation_section(img)
        elif mode == "TP7 - CC Labeling":
            process_tp7_section(img)
        elif mode == "TP8 - Region Props & Codes":
            process_tp8_section(img)
    else:
        st.info("Please upload an image.")
else:
    process_video_section()