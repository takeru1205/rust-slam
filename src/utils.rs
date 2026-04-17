/*
 *
 * to Convert from OpenCV 4xN Mat to 3D glam Vec3
 *
 */

use anyhow::{bail, Result};
use opencv::{core, prelude::*};


fn mat4xn_to_points3d(mat: &core::Mat) -> Result<Vec<[f32; 3]>> {
    let rows = mat.rows();
    let cols = mat.cols();

    if rows != 4 {
        bail!("triangulated_pts must be 4xN, but got {}x{}", rows, cols);
    }

    let typ = mat.typ();

    let mut points = Vec::with_capacity(cols as usize);

    match typ {
        t if t == core::CV_32F => {
            for c in 0..cols {
                let x = *mat.at_2d::<f32>(0, c)?;
                let y = *mat.at_2d::<f32>(1, c)?;
                let z = *mat.at_2d::<f32>(2, c)?;
                let w = *mat.at_2d::<f32>(3, c)?;

                if w.abs() > 1e-6 && x.is_finite() && y.is_finite() && z.is_finite() && w.is_finite()
                {
                    points.push([x / w, y / w, z / w]);
                }
            }
        }
        t if t == core::CV_64F => {
            for c in 0..cols {
                let x = *mat.at_2d::<f64>(0, c)?;
                let y = *mat.at_2d::<f64>(1, c)?;
                let z = *mat.at_2d::<f64>(2, c)?;
                let w = *mat.at_2d::<f64>(3, c)?;

                if w.abs() > 1e-9 && x.is_finite() && y.is_finite() && z.is_finite() && w.is_finite()
                {
                    points.push([(x / w) as f32, (y / w) as f32, (z / w) as f32]);
                }
            }
        }
        _ => {
            bail!("unsupported Mat type for triangulated_pts: {}", typ);
        }
    }

    Ok(points)
}


pub fn mat4xn_to_glam_points(mat: &core::Mat) -> Result<Vec<rerun::external::glam::Vec3>> {
    let raw = mat4xn_to_points3d(mat)?;

    Ok(raw
        .into_iter()
        .map(|p| rerun::external::glam::Vec3::new(p[0], p[1], p[2]))
        .collect())
}


pub fn mat4xn_to_glam_points_with_move(
    mat: &core::Mat,
    rmat: &core::Mat,
    tvec: &core::Mat,
) -> Result<Vec<rerun::external::glam::Vec3>> {

    let raw = mat4xn_to_points3d(mat)?;

    // tvec を [sx, sy, sz] に取り出す
    let tx = *tvec.at_2d::<f64>(0, 0)? as f32;
    let ty = *tvec.at_2d::<f64>(1, 0)? as f32;
    let tz = *tvec.at_2d::<f64>(2, 0)? as f32;

    // 回転行列 R
    let r00 = *rmat.at_2d::<f64>(0, 0)? as f32;
    let r01 = *rmat.at_2d::<f64>(0, 1)? as f32;
    let r02 = *rmat.at_2d::<f64>(0, 2)? as f32;

    let r10 = *rmat.at_2d::<f64>(1, 0)? as f32;
    let r11 = *rmat.at_2d::<f64>(1, 1)? as f32;
    let r12 = *rmat.at_2d::<f64>(1, 2)? as f32;

    let r20 = *rmat.at_2d::<f64>(2, 0)? as f32;
    let r21 = *rmat.at_2d::<f64>(2, 1)? as f32;
    let r22 = *rmat.at_2d::<f64>(2, 2)? as f32;

    Ok(raw
        .into_iter()
        .map(|p| {
            let x = p[0];
            let y = p[1];
            let z = p[2];

            let x2 = r00 * x + r01 * y + r02 * z + tx;
            let y2 = r10 * x + r11 * y + r12 * z + ty;
            let z2 = r20 * x + r21 * y + r22 * z + tz;

            rerun::external::glam::Vec3::new(x2, y2, z2)
        })
        .collect())
}
