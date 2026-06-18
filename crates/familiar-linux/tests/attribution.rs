#[test]
fn attributes_our_own_loopback_socket_to_this_pid() {
    use std::io::Read;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let h = std::thread::spawn(move || {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut b = [0u8; 1];
        let _ = s.read(&mut b);
    });
    let (_srv, _peer) = listener.accept().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));

    // The accepted server socket's *local* port is `port`; attribute it.
    let pr = familiar_linux::attribution::attribute(port).expect("attributed");
    assert_eq!(pr.pid, std::process::id());
    drop(_srv);
    let _ = h.join();
}
