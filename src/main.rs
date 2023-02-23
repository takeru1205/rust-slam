use opencv::{calib3d, core, features2d, highgui, imgproc, prelude::*, videoio, viz};
mod feature_detect;
mod matching;
mod preprocess;
mod read;

fn run() -> opencv::Result<()> {
    // paramters
    let camera_matrix = core::Mat::eye(3, 3, core::CV_64F)?;

    // camera window
    let window = "camrera";
    highgui::named_window(window, 1)?;
    // 3d window
    let window_3d = "SLAM";
    let mut viz_3d = viz::Viz3d::new(window_3d)?;
    viz_3d.show_widget(
        &window_3d,
        &viz::Widget::from(viz::WCoordinateSystem::new(1.0)?),
        core::Affine3d::default(),
    )?;

    // read file
    let file_name = "test.mp4";
    let (mut cam, mut frame) = read::read_file(file_name)?;

    // load frame
    videoio::VideoCapture::read(&mut cam, &mut frame)?;
    // resize and convert to gray scale frame
    let (resized_frame, gray) = preprocess::preprocess(&frame)?;

    let frame_size = core::Size::new(960, 540);
    println!("{:?}", frame_size);
    let mut writer = videoio::VideoWriter::new(
        "outputs/output.mp4",
        videoio::VideoWriter::fourcc('h', '2', '6', '4')?,
        cam.get(videoio::CAP_PROP_FPS)?,
        frame_size,
        true,
    )?;

    println!(
        "cols: {}, rows: {}, channels, {}",
        &resized_frame.cols(),
        &resized_frame.rows(),
        &resized_frame.channels()
    );

    println!(
        "cols: {}, rows: {}, channels, {}",
        &gray.cols(),
        &gray.rows(),
        &gray.channels()
    );

    // feature extraction
    let (mut kps, mut desc) = feature_detect::feature_detect(&resized_frame, &gray)?;
    let mut next_frame = core::Mat::default();
    loop {
        videoio::VideoCapture::read(&mut cam, &mut next_frame)?;
        if next_frame.size()?.width > 0 {
            // resize and convert to gray scale frame
            let (next_resized_frame, next_gray) = preprocess::preprocess(&next_frame)?;

            // feature extraction
            let (next_kps, next_desc) =
                feature_detect::feature_detect(&next_resized_frame, &next_gray)?;

            // draw keypoints on gbr image
            let mut next_image_with_keypoints = Mat::default();
            features2d::draw_keypoints(
                &next_resized_frame,
                &next_kps,
                &mut next_image_with_keypoints,
                core::Scalar::from([255.0, 255.0, 0.0, 255.0]),
                features2d::DrawMatchesFlags::DEFAULT,
            )?;

            // knn matching with Lowe's  ratio test filtering
            let (pts, from_pts, to_pts) = matching::knnmatch(kps, &next_kps, desc, &next_desc)?;

            // draw matching lines
            imgproc::polylines(
                &mut next_image_with_keypoints,
                &pts,
                false,
                core::Scalar::from([200.0, 100.0, 100.0, 255.0]),
                2,
                8,
                0,
            )?;

            // find essential matrix
            let essential_mat = calib3d::find_essential_mat_matrix(
                &from_pts,
                &to_pts,
                &camera_matrix,
                calib3d::RANSAC,
                0.999,
                1.0,
                &mut core::no_array(),
            )?;

            // recover pose and triangulated
            let mut rvec = core::Mat::default();
            let mut tvec = core::Mat::default();
            let mut triangulated_pts = core::Mat::default();

            let recover_pose_triangulated = calib3d::recover_pose_triangulated(
                &essential_mat,
                &from_pts,
                &to_pts,
                &camera_matrix,
                &mut rvec,
                &mut tvec,
                20.0, // distance_threshold
                &mut core::no_array(),
                &mut triangulated_pts,
            )?;
            println!("{}", recover_pose_triangulated);
            println!("{:?}", triangulated_pts);

            println!("aaa");
            let wcld = viz::WCloud::new(
                &triangulated_pts,
                &core::Scalar::from([255.0, 255.0, 0.0, 255.0]),
            )?;
            println!("bbb");
            viz_3d.show_widget(
                &window_3d,
                &viz::Widget::from(wcld),
                core::Affine3d::default(),
            )?;
            println!("ccc");

            // image show
            highgui::imshow(window, &next_image_with_keypoints)?;

            writer.write(&next_image_with_keypoints)?;
            // key wait
            let key = highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                videoio::VideoCapture::release(&mut cam)?;
                break;
            }
            kps = next_kps;
            desc = next_desc;
        } else {
            println!("No more frames!");
            videoio::VideoCapture::release(&mut cam)?;
            break ();
        }
    }
    Ok(())
}

fn main() {
    run().unwrap()
}
