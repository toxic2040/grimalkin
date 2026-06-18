//! A blocking control client: connect to the daemon's control socket and issue
//! one request, read one response. The UI keeps one client and reuses it.
use crate::{ControlRequest, ControlResponse, recv, send};
use std::io::{self, BufReader};
use std::os::unix::net::UnixStream;
use std::path::Path;

pub struct ControlClient {
    stream: UnixStream,
    reader: BufReader<UnixStream>,
}

impl ControlClient {
    pub fn connect(path: &Path) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self { stream, reader })
    }

    /// Send a request and block for the single response line.
    pub fn request(&mut self, req: &ControlRequest) -> io::Result<ControlResponse> {
        send(&mut self.stream, req)?;
        recv(&mut self.reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{recv, send};
    use std::io::BufReader;
    use std::os::unix::net::UnixListener;

    #[test]
    fn client_request_round_trips_over_a_unix_socket() {
        let dir = std::env::temp_dir().join(format!("fam-ipc-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sock = dir.join("t.sock");
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();

        // A one-shot echo server: read a request, reply Ok.
        let h = std::thread::spawn(move || {
            let (conn, _) = listener.accept().unwrap();
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let _req: ControlRequest = recv(&mut r).unwrap();
            let mut w = conn;
            send(&mut w, &ControlResponse::Ok).unwrap();
        });

        let mut client = ControlClient::connect(&sock).unwrap();
        let resp = client.request(&ControlRequest::GetStatus).unwrap();
        assert_eq!(resp, ControlResponse::Ok);
        h.join().unwrap();
    }
}
