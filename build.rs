fn main() {
    println!("cargo:rustc-link-lib=opencv_calib3d");
    println!("cargo:rustc-link-lib=opencv_features2d");
    // println!("cargo:rustc-link-lib=opencv_viz");
}
