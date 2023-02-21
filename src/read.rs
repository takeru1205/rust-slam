use opencv::{core, prelude::*, videoio};

pub fn read_file(file_name: &str) -> opencv::Result<(videoio::VideoCapture, core::Mat)> {
    // read file
    let mut cam = videoio::VideoCapture::from_file(&file_name, videoio::CAP_ANY)?;
    let opened_file = videoio::VideoCapture::open_file(&mut cam, &file_name, videoio::CAP_ANY)?;
    if !opened_file {
        panic!("Unable to open video file!");
    };
    let mut frame = core::Mat::default();
    let frame_read = videoio::VideoCapture::read(&mut cam, &mut frame)?;
    if !frame_read {
        panic!("Unable to read from video file!");
    };
    let opened = videoio::VideoCapture::is_opened(&mut cam)?;
    println!("Opened? {}", opened);
    if !opened {
        panic!("Unable to open video file!");
    };
    Ok((cam, frame))
}
