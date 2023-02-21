use opencv::core::Size;
use opencv::{core, imgproc, prelude::*};

pub fn preprocess(frame: &core::Mat) -> opencv::Result<(core::Mat, core::Mat)> {
    let mut resized_frame = Mat::default();
    imgproc::resize(&frame, &mut resized_frame, Size::default(), 0.5, 0.5, 1)?;
    // convert to gray scale
    let mut gray = Mat::default();
    imgproc::cvt_color(&resized_frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    Ok((resized_frame, gray))
}
