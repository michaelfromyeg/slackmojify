use anyhow::{Context, Result};
use std::{thread, time};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Data {
    // The string pattern to search for
    pattern: String,
    // The file path
    #[structopt(parse(from_os_str))]
    path: std::path::PathBuf, // a string but for file paths that can be supported across systems
}

// Not currently used; anyhow used instead
#[derive(Debug)]
struct CustomError(String);

fn main() -> Result<()> {
    println!("From Slackmojify: Hello, world!");

    let args = Data::from_args();
    let path_string = args.path.as_path().display().to_string();

    // Print the pattern and PathBuf
    println!("Pattern: {}", args.pattern);
    // https://stackoverflow.com/questions/37388107/how-to-convert-the-pathbuf-to-string/42579588
    println!("Path: {}", path_string);

    progress();

    let content = std::fs::read_to_string(path_string).with_context(|| {
        format!(
            "Could not read file `{}`",
            args.path.as_path().display().to_string() // can't re-use value
        )
    })?;

    // Print out lines with pattern
    for line in content.lines() {
        if line.contains(&args.pattern) {
            println!("{}", line);
        }
    }

    return Ok(());
}

fn progress() {
    let pb = indicatif::ProgressBar::new(100);
    for i in 0..100 {
        let ten_millis = time::Duration::from_millis(10);
        thread::sleep(ten_millis);
        pb.inc(1);
    }
    pb.finish_with_message("done");
}
