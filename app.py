"""
Computer Vision Toolkit - Streamlit App
---------------------------------------
1. Edge detection (Canny, Sobel, Scharr, Prewitt, Laplacian) + explanation
2. Keypoints & descriptors (SIFT / ORB)
3. Descriptor matching between two related images (with visualisation)
4. Feature classification with k-NN or SVM
5. Accuracy, precision, recall, F1-score, confusion matrix

Run locally:  streamlit run app.py
"""

import io
import os
import zipfile

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from skimage.feature import hog as skimage_hog
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_digits, load_sample_images
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Computer Vision Toolkit", page_icon="🔍", layout="wide")

MAX_DIM = 900
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def bgr2rgb(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def limit_size(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def decode_upload(file):
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupted image.")
    return limit_size(img)


@st.cache_data(show_spinner=False)
def sample_images():
    """Two sample photos bundled with scikit-learn (no internet needed)."""
    imgs = load_sample_images().images
    return [limit_size(cv2.cvtColor(np.ascontiguousarray(i), cv2.COLOR_RGB2BGR)) for i in imgs]


def pick_image(label, key, fallback):
    f = st.file_uploader(label, type=[e.strip(".") for e in IMG_EXT], key=key)
    if f is not None:
        try:
            return decode_upload(f)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the image: {exc}")
    st.caption("Using a built-in sample image. Upload your own to override.")
    return fallback()


def make_related(img):
    """Create a rotated + scaled + brightness-changed version of an image."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 0.8)
    out = cv2.warpAffine(img, m, (w, h))
    return cv2.convertScaleAbs(out, alpha=1.1, beta=-10)


def make_detector(name, n_features):
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=n_features)
    return cv2.ORB_create(nfeatures=n_features)


def png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


# --------------------------------------------------------------------------
# Tab 1: Edge detection
# --------------------------------------------------------------------------
EDGE_INFO = {
    "Canny": (
        "Canny smooths the image with a Gaussian filter, computes intensity gradients, "
        "thins them with non-maximum suppression and finally keeps edges using two thresholds "
        "(hysteresis). Result: thin, clean, well-connected edges. Lower thresholds -> more "
        "(noisier) edges, higher thresholds -> only strong edges."
    ),
    "Sobel": (
        "Sobel convolves the image with two 3x3 (or larger) derivative kernels to estimate "
        "horizontal and vertical gradients; the edge strength is the gradient magnitude. "
        "Result: thicker edges that are fairly robust to noise."
    ),
    "Scharr": (
        "Scharr is a more rotationally accurate variant of Sobel (3x3 kernels with better "
        "weights). It responds more precisely to fine edges and diagonal structures."
    ),
    "Prewitt": (
        "Prewitt is similar to Sobel but uses uniform weights in the kernel, giving a simple "
        "gradient estimate. It is a bit more sensitive to noise than Sobel."
    ),
    "Laplacian": (
        "Laplacian is a second-derivative operator. It highlights regions where intensity "
        "changes rapidly in all directions and is very sensitive to noise, so a Gaussian blur "
        "beforehand is recommended."
    ),
}


def edge_tab():
    st.subheader("1. Edge Detection")
    img = pick_image("Upload an image", "edge_img", lambda: sample_images()[1].copy())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    col_ctrl, col_out = st.columns([1, 3])
    with col_ctrl:
        method = st.selectbox("Method", list(EDGE_INFO.keys()))
        blur = st.slider("Gaussian blur kernel (odd)", 1, 15, 5, step=2)
        g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray

        if method == "Canny":
            lo = st.slider("Lower threshold", 0, 255, 80)
            hi = st.slider("Upper threshold", 0, 255, 160)
            if lo >= hi:
                st.warning("Lower threshold should be smaller than upper threshold.")
        else:
            thr = st.slider("Binary threshold (for edge map)", 0, 255, 60)
            ksize = 3
            if method == "Sobel":
                ksize = st.select_slider("Sobel kernel size", options=[1, 3, 5, 7], value=3)

    if method == "Canny":
        edges = cv2.Canny(g, lo, hi)
        magnitude, binary = None, edges
    else:
        gf = g.astype(np.float64)
        if method == "Sobel":
            gx = cv2.Sobel(gf, cv2.CV_64F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(gf, cv2.CV_64F, 0, 1, ksize=ksize)
            mag = cv2.magnitude(gx, gy)
        elif method == "Scharr":
            gx = cv2.Scharr(gf, cv2.CV_64F, 1, 0)
            gy = cv2.Scharr(gf, cv2.CV_64F, 0, 1)
            mag = cv2.magnitude(gx, gy)
        elif method == "Prewitt":
            kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
            ky = kx.T
            gx = cv2.filter2D(gf, cv2.CV_64F, kx)
            gy = cv2.filter2D(gf, cv2.CV_64F, ky)
            mag = cv2.magnitude(gx, gy)
        else:  # Laplacian
            mag = np.abs(cv2.Laplacian(gf, cv2.CV_64F, ksize=3))
        magnitude = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        binary = ((magnitude > thr) * 255).astype(np.uint8)

    with col_out:
        if magnitude is None:
            c1, c2 = st.columns(2)
            c1.image(bgr2rgb(img), caption="Original", use_container_width=True)
            c2.image(binary, caption="Canny edges", use_container_width=True)
        else:
            c1, c2, c3 = st.columns(3)
            c1.image(bgr2rgb(img), caption="Original", use_container_width=True)
            c2.image(magnitude, caption=f"{method} gradient magnitude", use_container_width=True)
            c3.image(binary, caption="Thresholded edge map", use_container_width=True)

    ratio = float(np.count_nonzero(binary)) / binary.size * 100
    st.markdown("#### Explanation of the result")
    st.write(EDGE_INFO[method])
    st.info(
        f"**Result:** {ratio:.2f}% of the pixels are marked as edges with the current "
        f"settings. Edges appear where brightness changes sharply (object boundaries, texture, "
        f"text). Increase the blur or thresholds to suppress noise; decrease them to reveal "
        f"finer detail."
    )
    st.download_button("Download edge map (PNG)", png_bytes(binary), "edges.png", "image/png")


# --------------------------------------------------------------------------
# Tab 2: Keypoints + descriptors
# --------------------------------------------------------------------------
def keypoints_tab():
    st.subheader("2. Keypoints & Descriptors (SIFT / ORB)")
    img = pick_image("Upload an image", "kp_img", lambda: sample_images()[0].copy())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    c1, c2 = st.columns(2)
    algo = c1.radio("Algorithm", ["SIFT", "ORB"], horizontal=True, key="kp_algo")
    n_feat = c2.slider("Max number of features", 50, 3000, 500, step=50, key="kp_n")

    det = make_detector(algo, n_feat)
    kps, des = det.detectAndCompute(gray, None)

    if des is None or len(kps) == 0:
        st.warning("No keypoints found. Try another image.")
        return

    vis = cv2.drawKeypoints(
        img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0)
    )
    a, b = st.columns(2)
    a.image(bgr2rgb(img), caption="Original", use_container_width=True)
    b.image(bgr2rgb(vis), caption=f"{algo} keypoints (circle size = scale, line = orientation)",
            use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Keypoints detected", len(kps))
    m2.metric("Descriptor shape", f"{des.shape[0]} x {des.shape[1]}")
    m3.metric("Descriptor type", f"{des.dtype}")

    st.markdown("#### Explanation")
    if algo == "SIFT":
        st.write(
            "SIFT finds scale- and rotation-invariant blobs/corners using Difference-of-Gaussians "
            "and describes each with a 128-dimensional float vector (histograms of gradient "
            "orientations). It is accurate and robust but slower."
        )
    else:
        st.write(
            "ORB combines the FAST corner detector with a rotation-aware BRIEF binary descriptor "
            "(256 bits = 32 bytes). It is very fast and free to use, but a little less robust "
            "to large scale changes than SIFT."
        )
    with st.expander("Show first 5 descriptors"):
        st.dataframe(pd.DataFrame(des[:5]))


# --------------------------------------------------------------------------
# Tab 3: Matching
# --------------------------------------------------------------------------
def matching_tab():
    st.subheader("3. Descriptor Matching")
    left, right = st.columns(2)
    with left:
        img1 = pick_image("Image 1", "m_img1", lambda: sample_images()[0].copy())
    with right:
        img2 = pick_image("Image 2 (related to image 1)", "m_img2", lambda: make_related(img1))

    c1, c2, c3, c4 = st.columns(4)
    algo = c1.radio("Algorithm", ["SIFT", "ORB"], horizontal=True, key="m_algo")
    n_feat = c2.slider("Max features", 100, 4000, 1000, step=100, key="m_n")
    ratio = c3.slider("Lowe ratio test", 0.50, 0.95, 0.75, step=0.01)
    top_n = c4.slider("Matches to draw", 10, 200, 50, step=5)
    use_ransac = st.checkbox("Filter matches with RANSAC homography (show only inliers)", value=True)

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    det = make_detector(algo, n_feat)
    kp1, des1 = det.detectAndCompute(g1, None)
    kp2, des2 = det.detectAndCompute(g2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        st.warning("Not enough keypoints in one of the images.")
        return

    norm = cv2.NORM_L2 if algo == "SIFT" else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    good = sorted(good, key=lambda m: m.distance)

    mask_list = None
    inliers = None
    if use_ransac and len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            mask_list = mask.ravel().tolist()
            inliers = int(mask.sum())

    shown = good[:top_n]
    mask_shown = mask_list[:top_n] if mask_list is not None else None
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, shown, None,
        matchesMask=mask_shown,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    st.image(bgr2rgb(vis), caption=f"{algo} matches", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Keypoints (img 1)", len(kp1))
    m2.metric("Keypoints (img 2)", len(kp2))
    m3.metric("Good matches", len(good))
    m4.metric("RANSAC inliers", inliers if inliers is not None else "-")

    st.markdown("#### Explanation")
    st.write(
        "Each descriptor of image 1 is compared with all descriptors of image 2 using a "
        f"brute-force matcher ({'L2 distance' if algo == 'SIFT' else 'Hamming distance'}). "
        "Lowe's ratio test keeps a match only if the best candidate is clearly better than the "
        "second best. RANSAC then removes matches that do not agree with a single geometric "
        "transform (homography), leaving reliable correspondences."
    )


# --------------------------------------------------------------------------
# Tab 4: Classification
# --------------------------------------------------------------------------
def hog_feature(img):
    g = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY)
    return skimage_hog(
        g,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def color_hist_feature(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h)
    return h.flatten()


def local_descriptors(img, algo):
    g = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2GRAY)
    if algo == "SIFT":
        det = cv2.SIFT_create()
    else:
        det = cv2.ORB_create(nfeatures=300, edgeThreshold=15, patchSize=15)
    _, des = det.detectAndCompute(g, None)
    return None if des is None else des.astype(np.float32)


def bovw_features(train_imgs, test_imgs, algo, k):
    """Bag of Visual Words: vocabulary learnt on the TRAIN images only."""
    tr_des = [local_descriptors(i, algo) for i in train_imgs]
    te_des = [local_descriptors(i, algo) for i in test_imgs]
    valid = [d for d in tr_des if d is not None]
    if not valid:
        raise ValueError("No keypoints could be detected in the training images.")
    stack = np.vstack(valid)
    rng = np.random.default_rng(42)
    if len(stack) > 50000:
        stack = stack[rng.choice(len(stack), 50000, replace=False)]
    k = max(2, min(k, len(stack)))
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024).fit(stack)

    def hist(des):
        h = np.zeros(k, dtype=np.float64)
        if des is not None:
            h = np.bincount(km.predict(des), minlength=k).astype(np.float64)
            s = h.sum()
            if s > 0:
                h /= s
        return h

    return np.array([hist(d) for d in tr_des]), np.array([hist(d) for d in te_des])


@st.cache_data(show_spinner=False)
def load_digits_data():
    d = load_digits()
    imgs = []
    for im in d.images:
        im8 = (im / 16.0 * 255).astype(np.uint8)
        im8 = cv2.resize(im8, (64, 64), interpolation=cv2.INTER_CUBIC)
        imgs.append(cv2.cvtColor(im8, cv2.COLOR_GRAY2BGR))
    return imgs, [str(t) for t in d.target]


@st.cache_data(show_spinner=False)
def load_zip_data(zip_bytes):
    imgs, labels = [], []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if name.endswith("/") or "__MACOSX" in name or base.startswith("."):
                continue
            if not name.lower().endswith(IMG_EXT):
                continue
            parts = [p for p in name.split("/") if p]
            if len(parts) < 2:
                continue
            data = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            imgs.append(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA))
            labels.append(parts[-2])
    return imgs, labels


def classification_tab():
    st.subheader("4. Feature Classification (k-NN / SVM) + Metrics")

    source = st.radio(
        "Dataset",
        ["Built-in digits dataset (quick demo)", "Upload ZIP (one folder per class)"],
        horizontal=True,
    )
    imgs, labels = None, None
    if source.startswith("Built-in"):
        imgs, labels = load_digits_data()
        st.caption("scikit-learn digits: 1797 images, 10 classes (0-9).")
    else:
        st.caption(
            "ZIP structure: `dataset.zip/cat/img1.jpg`, `dataset.zip/dog/img2.jpg`, ... "
            "(folder name = class label, at least 2 classes, ~20+ images per class recommended)."
        )
        zf = st.file_uploader("Upload dataset ZIP", type=["zip"], key="zip_up")
        if zf is not None:
            try:
                imgs, labels = load_zip_data(zf.getvalue())
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read ZIP: {exc}")
        if not imgs:
            st.info("Upload a ZIP to continue.")
            return

    labels_arr = np.array(labels)
    classes, counts = np.unique(labels_arr, return_counts=True)
    st.write(f"**{len(imgs)} images, {len(classes)} classes:** "
             + ", ".join(f"{c} ({n})" for c, n in zip(classes, counts)))
    if len(classes) < 2:
        st.error("At least 2 classes are required.")
        return

    c1, c2, c3 = st.columns(3)
    feat_name = c1.selectbox(
        "Feature type",
        ["HOG", "Color histogram (HSV)", "Bag of Visual Words - SIFT", "Bag of Visual Words - ORB"],
    )
    clf_name = c2.selectbox("Classifier", ["k-NN", "SVM"])
    test_size = c3.slider("Test size", 0.1, 0.5, 0.25, step=0.05)

    d1, d2, d3 = st.columns(3)
    vocab = 50
    if feat_name.startswith("Bag"):
        vocab = d1.slider("Vocabulary size (k-means clusters)", 10, 300, 50, step=10)
    if clf_name == "k-NN":
        k_nn = d2.slider("k (neighbours)", 1, 25, 5)
    else:
        kernel = d2.selectbox("SVM kernel", ["rbf", "linear", "poly"])
        c_val = d3.slider("C", 0.1, 100.0, 10.0)

    if not st.button("Extract features, train & evaluate", type="primary"):
        return

    idx = np.arange(len(imgs))
    stratify = labels_arr if counts.min() >= 2 else None
    try:
        tr_idx, te_idx = train_test_split(
            idx, test_size=test_size, random_state=42, stratify=stratify
        )
    except ValueError as exc:
        st.error(f"Split failed: {exc}")
        return
    y_tr, y_te = labels_arr[tr_idx], labels_arr[te_idx]
    tr_imgs = [imgs[i] for i in tr_idx]
    te_imgs = [imgs[i] for i in te_idx]

    with st.spinner("Extracting features..."):
        try:
            if feat_name == "HOG":
                x_tr = np.array([hog_feature(i) for i in tr_imgs])
                x_te = np.array([hog_feature(i) for i in te_imgs])
            elif feat_name.startswith("Color"):
                x_tr = np.array([color_hist_feature(i) for i in tr_imgs])
                x_te = np.array([color_hist_feature(i) for i in te_imgs])
            else:
                algo = "SIFT" if "SIFT" in feat_name else "ORB"
                x_tr, x_te = bovw_features(tr_imgs, te_imgs, algo, vocab)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Feature extraction failed: {exc}")
            return

    if clf_name == "k-NN":
        model = make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=min(k_nn, len(y_tr)))
        )
    else:
        model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c_val, gamma="scale"))

    with st.spinner("Training..."):
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)

    acc = accuracy_score(y_te, pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_te, pred, average="weighted", zero_division=0
    )

    st.success(f"Trained on {len(y_tr)} images, evaluated on {len(y_te)} images "
               f"({x_tr.shape[1]} features per image).")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision (macro)", f"{p_m:.4f}")
    m3.metric("Recall (macro)", f"{r_m:.4f}")
    m4.metric("F1-score (macro)", f"{f_m:.4f}")

    summary = pd.DataFrame(
        {
            "Precision": [p_m, p_w],
            "Recall": [r_m, r_w],
            "F1-score": [f_m, f_w],
        },
        index=["Macro average", "Weighted average"],
    ).round(4)
    st.dataframe(summary, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Per-class report")
        rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)
    with right:
        st.markdown("#### Confusion matrix")
        labs = sorted(np.unique(np.concatenate([y_te, pred])))
        cm = confusion_matrix(y_te, pred, labels=labs)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labs)))
        ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=45, ha="right")
        ax.set_yticklabels(labs)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        if len(labs) <= 15:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Explanation")
    st.write(
        "**Accuracy** = correct predictions / all predictions. **Precision** = of the images "
        "predicted as a class, how many truly belong to it. **Recall** = of the images that truly "
        "belong to a class, how many were found. **F1-score** = harmonic mean of precision and "
        "recall. Macro average treats all classes equally; weighted average accounts for class "
        "sizes."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.title("🔍 Computer Vision Toolkit")
    st.caption("Edge detection • SIFT/ORB features • Descriptor matching • k-NN/SVM classification")

    t1, t2, t3, t4 = st.tabs(
        ["Edge Detection", "Keypoints & Descriptors", "Feature Matching", "Classification & Metrics"]
    )
    with t1:
        edge_tab()
    with t2:
        keypoints_tab()
    with t3:
        matching_tab()
    with t4:
        classification_tab()


if __name__ == "__main__":
    main()
