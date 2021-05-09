use anyhow::{Context, Result};
// use futures::executor::block_on;
use image::GenericImageView;
use log::info;
// use serde::{Deserialize, Serialize};
use std::{error, path, thread, time};
use structopt::StructOpt;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};

#[derive(Debug, StructOpt)]
struct Data {
    // The string pattern to search for
    pattern: String,
    // The file path
    #[structopt(parse(from_os_str))]
    input: path::PathBuf, // a string but for file paths that can be supported across systems
    #[structopt(parse(from_os_str))]
    output: path::PathBuf,
    #[structopt(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

#[derive(Copy, Clone, Debug)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
}

const LINE_COLOUR: image::Rgba<u8> = image::Rgba { 0: [0, 255, 0, 0] };

// Not currently used; anyhow used instead
#[derive(Debug)]
struct CustomError(String);

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    info!("Starting up...");

    println!("From Slackmojify: Hello, world!");

    let args = Data::from_args();

    // Print the pattern and PathBuf
    println!("Pattern: {}", args.pattern);
    // https://stackoverflow.com/questions/37388107/how-to-convert-the-pathbuf-to-string/42579588
    println!("Path: {}", args.input.as_path().display().to_string());
    println!("Verbosity: {}", args.verbose);

    // grep_file(args.input, args.pattern);

    // progress();

    // detect_faces(
    //     args.input.as_path().display().to_string(),
    //     args.output.as_path().display().to_string(),
    // )
    // .expect("face detection error");

    let _res = send_to_slack().await;
    // let _res = block_on(promise);

    info!("Done!");

    return Ok(());
}

#[allow(dead_code)]
fn progress() {
    let pb = indicatif::ProgressBar::new(100);
    for _i in 0..100 {
        let ten_millis = time::Duration::from_millis(10);
        thread::sleep(ten_millis);
        pb.inc(1);
    }
    pb.finish_with_message("done");
}

#[allow(dead_code)]
fn detect_faces(input: String, output: String) -> Result<(), Box<dyn error::Error>> {
    println!("input: {}", input);
    println!("output: {}", output);

    let model = include_bytes!("mtcnn.pb");
    let mut graph = Graph::new();
    graph.import_graph_def(&*model, &ImportGraphDefOptions::new())?;

    let input_image = image::open(&input)?;

    let mut flattened: Vec<f32> = Vec::new();

    for (_x, _y, rgb) in input_image.pixels() {
        flattened.push(rgb[2] as f32);
        flattened.push(rgb[1] as f32);
        flattened.push(rgb[0] as f32);
    }

    let input = Tensor::new(&[input_image.height() as u64, input_image.width() as u64, 3])
        .with_values(&flattened)?;

    let mut session = Session::new(&SessionOptions::new(), &graph)?;

    let min_size = Tensor::new(&[]).with_values(&[40f32])?;
    let thresholds = Tensor::new(&[3]).with_values(&[0.6f32, 0.7f32, 0.7f32])?;
    let factor = Tensor::new(&[]).with_values(&[0.709f32])?;

    let mut args = SessionRunArgs::new();

    // Load our parameters for the model
    args.add_feed(&graph.operation_by_name_required("min_size")?, 0, &min_size);
    args.add_feed(
        &graph.operation_by_name_required("thresholds")?,
        0,
        &thresholds,
    );
    args.add_feed(&graph.operation_by_name_required("factor")?, 0, &factor);

    // Load our input image
    args.add_feed(&graph.operation_by_name_required("input")?, 0, &input);

    let bbox = args.request_fetch(&graph.operation_by_name_required("box")?, 0);
    let prob = args.request_fetch(&graph.operation_by_name_required("prob")?, 0);

    session.run(&mut args)?;

    let bbox_res: Tensor<f32> = args.fetch(bbox)?;
    let prob_res: Tensor<f32> = args.fetch(prob)?;

    let bboxes: Vec<_> = bbox_res
        .chunks_exact(4) // Split into chunks of 4
        .zip(prob_res.iter()) // Combine it with prob_res
        .map(|(bbox, &prob)| BBox {
            y1: bbox[0],
            x1: bbox[1],
            y2: bbox[2],
            x2: bbox[3],
            prob,
        })
        .collect();

    println!("BBox Length: {}, Bboxes:{:#?}", bboxes.len(), bboxes);

    let mut output_image = input_image;

    for bbox in bboxes {
        let rect = imageproc::rect::Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);
        imageproc::drawing::draw_hollow_rect_mut(&mut output_image, rect, LINE_COLOUR);
        let cropped_image = image::imageops::crop(
            &mut output_image,
            bbox.x1 as u32,
            bbox.y1 as u32,
            (bbox.x2 - bbox.x1) as u32,
            (bbox.y2 - bbox.y1) as u32,
        );
        cropped_image
            .to_image()
            .save_with_format("test_crop.jpg", image::ImageFormat::Jpeg)
            .unwrap();
    }

    output_image.save(&output)?;

    // Now produce crop

    return Ok(());
}

#[allow(dead_code)]
fn grep_file(input: path::PathBuf, pattern: String) -> Result<()> {
    let content =
        std::fs::read_to_string(input.as_path().display().to_string()).with_context(|| {
            format!(
                "Could not read file `{}`",
                input.as_path().display().to_string() // can't re-use value
            )
        })?;

    // Print out lines with pattern
    for line in content.lines() {
        if line.contains(&pattern) {
            println!("{}", line);
        }
    }

    return Ok(());
}

async fn send_to_slack() -> Result<()> {
    let user_token = "Bearer ...";
    let url = "https://slack.com/api/emoji.list";

    println!("send_to_slack");

    let client = reqwest::Client::new();
    let res = client
        .get(url)
        .header("Authorization", user_token)
        .send()
        .await?
        .text()
        .await?;

    let data: serde_json::Value = serde_json::from_str(&res).unwrap();

    println!("data = {:?}", data);

    return Ok(());
}
