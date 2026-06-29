//! The file-read source: a background thread that accepts the privileged
//! fanotify helper's connection on a Unix socket and streams its newline-JSON
//! FileReadEvents onto a channel. In tests, a plain channel stands in for the
//! socket so the network path is exercisable without the helper.
use familiar_linux::wire::FileReadEvent;
use std::io::{self, BufRead, BufReader, Read};
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};

const MAX_FILE_READ_EVENT_BYTES: u64 = 64 * 1024;

fn read_helper_line<R: BufRead>(reader: &mut R) -> io::Result<Option<String>> {
    let mut line = String::new();
    let n = reader
        .take(MAX_FILE_READ_EVENT_BYTES + 1)
        .read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }
    if n as u64 > MAX_FILE_READ_EVENT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file-read event frame too large",
        ));
    }
    Ok(Some(line))
}

/// Bind the helper socket, accept helper connections, and stream their
/// newline-delimited FileReadEvent JSON onto a channel. The systemd units
/// restrict /run/familiar to the daemon + helper.
pub fn spawn_socket_source(
    socket: &Path,
) -> std::io::Result<(Receiver<FileReadEvent>, JoinHandle<()>)> {
    let (tx, rx) = channel();
    let handle = spawn_socket_source_to(socket, tx)?;
    Ok((rx, handle))
}

/// Bind the helper socket using a caller-supplied channel. This lets the daemon
/// build its supervisor while deferring the actual sensor socket until armed.
pub fn spawn_socket_source_to(
    socket: &Path,
    tx: Sender<FileReadEvent>,
) -> std::io::Result<JoinHandle<()>> {
    if socket.exists() {
        let _ = std::fs::remove_file(socket);
    }
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(socket)?;
    let handle = thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            // F8: the only legitimate client is the CAP_SYS_ADMIN helper (root).
            // Reject anything else — a non-root process must not be able to spoof
            // FileRead events or hold the socket to starve the real helper.
            match rustix::net::sockopt::get_socket_peercred(&stream) {
                Ok(cred) if cred.uid.as_raw() == 0 => {}
                Ok(cred) => {
                    eprintln!(
                        "[familiar] fileread: rejecting non-root peer uid {}",
                        cred.uid.as_raw()
                    );
                    continue;
                }
                Err(e) => {
                    eprintln!("[familiar] fileread: cannot read peer cred: {e}");
                    continue;
                }
            }
            let mut reader = BufReader::new(stream);
            loop {
                let line = match read_helper_line(&mut reader) {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("[familiar] fileread: closing helper stream: {e}");
                        break;
                    }
                };
                if let Ok(ev) = serde_json::from_str::<FileReadEvent>(&line)
                    && tx.send(ev).is_err()
                {
                    return; // daemon gone
                }
            }
        }
    });
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn helper_line_cap_rejects_oversized_events() {
        let mut cur = Cursor::new(vec![b'a'; (MAX_FILE_READ_EVENT_BYTES + 1) as usize]);
        let err = read_helper_line(&mut cur).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
