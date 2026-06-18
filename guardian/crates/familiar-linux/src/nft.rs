//! The dedicated `inet familiar` nftables table and the reversible drop rule.
//! Every nft operation goes through the `nft` userspace binary (no netlink
//! library), keeping the crate dependency-light and free of copyleft deps.
//! Per-block reversal removes a single rule by handle; `reverse_all` flushes the
//! block chain; `delete_table` is the full-teardown primitive.
use std::net::Ipv4Addr;

pub const TABLE: &str = "familiar";
pub const BLOCK_CHAIN: &str = "egress-block";
pub const SENSE_CHAIN: &str = "egress-sense";

#[derive(Debug, thiserror::Error)]
pub enum NftError {
    #[error("netlink send failed: {0}")]
    Send(String),
}

use std::process::Command;

/// Run `nft <args...>` and return stdout, mapping any failure to `NftError::Send`.
/// (Argument form only; `install_queue_rule` keeps its `-f -` stdin pipe.)
fn run_nft(args: &[&str]) -> Result<String, NftError> {
    let out = Command::new("nft")
        .args(args)
        .output()
        .map_err(|e| NftError::Send(format!("spawn nft: {e}")))?;
    if !out.status.success() {
        return Err(NftError::Send(
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Remove every DROP rule from the block chain while leaving the table and the
/// `egress-sense` NFQUEUE chain in place — so lifting containment never blinds
/// the sensor (the divert rule is installed only at startup).
pub fn flush_block_chain() -> Result<(), NftError> {
    run_nft(&["flush", "chain", "inet", TABLE, BLOCK_CHAIN]).map(|_| ())
}

/// Create the dedicated `inet familiar` table and the block chain. Idempotent:
/// `nft add` of an existing table/chain is accepted.
pub fn ensure_table() -> Result<(), NftError> {
    run_nft(&["add", "table", "inet", TABLE])?;
    run_nft(&[
        "add",
        "chain",
        "inet",
        TABLE,
        BLOCK_CHAIN,
        "{ type filter hook output priority 0; policy accept; }",
    ])?;
    Ok(())
}

/// Install the sense chain: divert all outbound TCP to NFQUEUE `queue_num`. The
/// userspace reader ACCEPTs each packet; the SYN filter that turns this into
/// new-connection sensing lives in the parser.
///
/// Uses `nft -f -` (stdin pipe) because the `queue` statement must be written as
/// a single batch. The chain is added to the same dedicated `familiar` table, so
/// `delete_table` still reverses everything in one operation.
pub fn install_queue_rule(queue_num: u16) -> Result<(), NftError> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let batch = format!(
        "add chain inet {TABLE} {SENSE_CHAIN} {{ type filter hook output priority 1; policy accept; }}\n\
         add rule inet {TABLE} {SENSE_CHAIN} meta l4proto tcp queue num {queue_num}\n"
    );
    let mut child = Command::new("nft")
        .args(["-f", "-"])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| NftError::Send(format!("spawn nft: {e}")))?;
    child
        .stdin
        .take()
        .ok_or_else(|| NftError::Send("nft stdin".into()))?
        .write_all(batch.as_bytes())
        .map_err(|e| NftError::Send(e.to_string()))?;
    let out = child
        .wait_with_output()
        .map_err(|e| NftError::Send(e.to_string()))?;
    if !out.status.success() {
        return Err(NftError::Send(
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ));
    }
    Ok(())
}

/// Install a reversible DROP for outbound TCP to `dst:dport`. Returns a note for
/// the audit/notify trail. The rule renders as `ip daddr <dst> tcp dport <dport>
/// drop`, which is what `parse_handle`/`unblock_outbound` match on.
pub fn block_outbound(dst: Ipv4Addr, dport: u16) -> Result<String, NftError> {
    let ip = dst.to_string();
    let port = dport.to_string();
    run_nft(&[
        "add",
        "rule",
        "inet",
        TABLE,
        BLOCK_CHAIN,
        "ip",
        "daddr",
        &ip,
        "tcp",
        "dport",
        &port,
        "drop",
    ])?;
    Ok(format!("nft drop {dst}:{dport} in table inet {TABLE}"))
}

/// Reverse everything by deleting the dedicated table.
pub fn delete_table() -> Result<(), NftError> {
    run_nft(&["delete", "table", "inet", TABLE]).map(|_| ())
}

/// Find the kernel handle of the DROP rule for `dst:dport` in a `nft -a list`
/// chain dump. Token-exact on `daddr`/`dport` so a port prefix cannot collide.
fn parse_handle(listing: &str, dst: Ipv4Addr, dport: u16) -> Option<u64> {
    let ip = dst.to_string();
    let port = dport.to_string();
    for line in listing.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        let pair = |k: &str, v: &str| toks.windows(2).any(|w| w[0] == k && w[1] == v);
        if pair("daddr", &ip) && pair("dport", &port) && toks.contains(&"drop") {
            // tokens: ... drop # handle N
            if let Some(i) = toks.iter().position(|t| *t == "handle") {
                return toks.get(i + 1).and_then(|n| n.parse().ok());
            }
        }
    }
    None
}

/// Remove exactly the reversible DROP rule for `dst:dport` (by kernel handle),
/// leaving every other block intact. Errors if no such rule exists.
pub fn unblock_outbound(dst: Ipv4Addr, dport: u16) -> Result<String, NftError> {
    let listing = run_nft(&["-a", "list", "chain", "inet", TABLE, BLOCK_CHAIN])?;
    let handle = parse_handle(&listing, dst, dport)
        .ok_or_else(|| NftError::Send(format!("no drop rule for {dst}:{dport}")))?;
    let h = handle.to_string();
    run_nft(&["delete", "rule", "inet", TABLE, BLOCK_CHAIN, "handle", &h])?;
    Ok(format!("removed nft drop {dst}:{dport} (handle {handle})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
table inet familiar {
\tchain egress-block {
\t\ttype filter hook output priority filter; policy accept;
\t\tip daddr 203.0.113.9 tcp dport 443 drop # handle 4
\t\tip daddr 198.51.100.4 tcp dport 8443 drop # handle 7
\t}
}";

    #[test]
    fn parse_handle_finds_the_matching_rule() {
        assert_eq!(
            parse_handle(SAMPLE, "203.0.113.9".parse().unwrap(), 443),
            Some(4)
        );
        assert_eq!(
            parse_handle(SAMPLE, "198.51.100.4".parse().unwrap(), 8443),
            Some(7)
        );
    }

    #[test]
    fn parse_handle_does_not_confuse_a_port_prefix() {
        // dport 443 must not match a rule for dport 4430.
        let listing = "\t\tip daddr 203.0.113.9 tcp dport 4430 drop # handle 9";
        assert_eq!(
            parse_handle(listing, "203.0.113.9".parse().unwrap(), 443),
            None
        );
    }

    #[test]
    fn parse_handle_returns_none_when_absent() {
        assert_eq!(parse_handle(SAMPLE, "10.0.0.1".parse().unwrap(), 443), None);
    }
}
