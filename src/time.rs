//! Datetime helpers. Datetime coordinates are stored like numpy's
//! `datetime64[ns]`: `i64` nanoseconds since 1970-01-01, with [`NAT`] as the
//! missing value.

use crate::error::{Error, Result};
use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime, Timelike};

/// "Not a time": the missing datetime value.
pub const NAT: i64 = i64::MIN;

const NS_PER_SEC: i64 = 1_000_000_000;

/// Converts a datetime to nanoseconds since the Unix epoch.
pub fn to_ns(dt: NaiveDateTime) -> i64 {
    let utc = dt.and_utc();
    utc.timestamp() * NS_PER_SEC + utc.timestamp_subsec_nanos() as i64
}

/// Converts nanoseconds since the Unix epoch back to a datetime (`None` for
/// [`NAT`]).
pub fn from_ns(ns: i64) -> Option<NaiveDateTime> {
    if ns == NAT {
        return None;
    }
    let secs = ns.div_euclid(NS_PER_SEC);
    let nanos = ns.rem_euclid(NS_PER_SEC) as u32;
    chrono::DateTime::from_timestamp(secs, nanos).map(|d| d.naive_utc())
}

/// Resolution of a datetime string or index, coarsest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Resolution {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Sub,
}

/// Parses an ISO-8601-like datetime string. Accepts `YYYY`, `YYYY-MM`,
/// `YYYY-MM-DD`, and `YYYY-MM-DD[T ]HH[:MM[:SS[.fff]]]`.
pub fn parse_datetime(s: &str) -> Result<NaiveDateTime> {
    parse_partial(s).map(|(start, _, _)| start)
}

/// Returns `(start, exclusive_end, resolution)` for a possibly partial string.
pub(crate) fn parse_partial(s: &str) -> Result<(NaiveDateTime, NaiveDateTime, Resolution)> {
    let bad = || Error::invalid(format!("cannot parse datetime {s:?}"));
    let s = s.trim().trim_end_matches('Z');
    let (date_part, time_part) = match s.find(['T', ' ']) {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    let parts: Vec<&str> = date_part.split('-').collect();
    let num = |p: &str| p.parse::<u32>().map_err(|_| bad());
    let year: i32 = parts.first().ok_or_else(bad)?.parse().map_err(|_| bad())?;
    let month = parts.get(1).map(|p| num(p)).transpose()?;
    let day = parts.get(2).map(|p| num(p)).transpose()?;
    if parts.len() > 3 {
        return Err(bad());
    }
    let date = NaiveDate::from_ymd_opt(year, month.unwrap_or(1), day.unwrap_or(1)).ok_or_else(bad)?;
    let midnight = date.and_hms_opt(0, 0, 0).unwrap();
    match (month, day, time_part) {
        (None, _, None) => {
            let end = NaiveDate::from_ymd_opt(year + 1, 1, 1).ok_or_else(bad)?;
            Ok((midnight, end.and_hms_opt(0, 0, 0).unwrap(), Resolution::Year))
        }
        (Some(_), None, None) => Ok((midnight, add_months(midnight, 1), Resolution::Month)),
        (Some(_), Some(_), None) => Ok((midnight, midnight + Duration::days(1), Resolution::Day)),
        (Some(_), Some(_), Some(t)) => {
            let tp: Vec<&str> = t.split(':').collect();
            let h = num(tp[0])?;
            let m = tp.get(1).map(|p| num(p)).transpose()?;
            let (sec, nanos, sub) = match tp.get(2) {
                None => (None, 0, false),
                Some(p) => {
                    let (whole, frac) = p.split_once('.').unwrap_or((p, ""));
                    let mut nanos = 0u32;
                    if !frac.is_empty() {
                        let digits: String = frac.chars().chain("000000000".chars()).take(9).collect();
                        nanos = digits.parse().map_err(|_| bad())?;
                    }
                    (Some(num(whole)?), nanos, !frac.is_empty())
                }
            };
            let start = date.and_hms_nano_opt(h, m.unwrap_or(0), sec.unwrap_or(0), nanos).ok_or_else(bad)?;
            let (end, res) = match (m, sec) {
                (None, _) => (start + Duration::hours(1), Resolution::Hour),
                (Some(_), None) => (start + Duration::minutes(1), Resolution::Minute),
                _ if sub => (start + Duration::nanoseconds(1), Resolution::Sub),
                _ => (start + Duration::seconds(1), Resolution::Second),
            };
            Ok((start, end, res))
        }
        _ => Err(bad()),
    }
}

/// Finest resolution needed to represent all values of an index.
pub(crate) fn index_resolution(values: &[i64]) -> Resolution {
    let mut res = Resolution::Day;
    for &v in values {
        if v == NAT {
            continue;
        }
        let r = v.rem_euclid(86_400 * NS_PER_SEC);
        let this = if r == 0 {
            Resolution::Day
        } else if r % (3600 * NS_PER_SEC) == 0 {
            Resolution::Hour
        } else if r % (60 * NS_PER_SEC) == 0 {
            Resolution::Minute
        } else if r % NS_PER_SEC == 0 {
            Resolution::Second
        } else {
            Resolution::Sub
        };
        res = res.max(this);
    }
    res
}

/// Adds calendar months, clamping the day of month.
pub(crate) fn add_months(dt: NaiveDateTime, months: i32) -> NaiveDateTime {
    let total = dt.year() * 12 + dt.month0() as i32 + months;
    let (y, m0) = (total.div_euclid(12), total.rem_euclid(12) as u32);
    let mut day = dt.day();
    loop {
        if let Some(d) = NaiveDate::from_ymd_opt(y, m0 + 1, day) {
            return d.and_time(dt.time());
        }
        day -= 1;
    }
}

/// Last day of the month containing `dt`, at midnight.
pub(crate) fn month_end(year: i32, month0: u32) -> NaiveDateTime {
    let first = NaiveDate::from_ymd_opt(year, month0 + 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    add_months(first, 1) - Duration::days(1)
}

/// Formats a nanosecond timestamp like numpy/pandas do in reprs.
pub(crate) fn format_ns(ns: i64, date_only: bool) -> String {
    match from_ns(ns) {
        None => "NaT".to_string(),
        Some(dt) if date_only => dt.format("%Y-%m-%d").to_string(),
        Some(dt) => {
            if dt.nanosecond() == 0 {
                dt.format("%Y-%m-%dT%H:%M:%S").to_string()
            } else {
                dt.format("%Y-%m-%dT%H:%M:%S%.9f").to_string()
            }
        }
    }
}

/// Formats a nanosecond timestamp as ISO-8601 (used for JSON output).
pub fn format_iso(ns: i64) -> String {
    format_ns(ns, false)
}

/// Calendar component of a timestamp used by `groupby("time.month")` and
/// friends. Returns the component as an integer (season returns 0..4 for
/// DJF, MAM, JJA, SON).
pub(crate) fn component(ns: i64, comp: &str) -> Result<i64> {
    let dt = from_ns(ns).ok_or_else(|| Error::invalid("NaT in datetime component"))?;
    Ok(match comp {
        "year" => dt.year() as i64,
        "month" => dt.month() as i64,
        "day" => dt.day() as i64,
        "hour" => dt.hour() as i64,
        "minute" => dt.minute() as i64,
        "second" => dt.second() as i64,
        "dayofyear" => dt.ordinal() as i64,
        "dayofweek" | "weekday" => dt.weekday().num_days_from_monday() as i64,
        "quarter" => (dt.month0() / 3 + 1) as i64,
        "season" => match dt.month() {
            12 | 1 | 2 => 0,
            3..=5 => 1,
            6..=8 => 2,
            _ => 3,
        },
        other => return Err(Error::invalid(format!("unknown datetime component {other:?}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dt = parse_datetime("2000-01-02T03:04:05.5").unwrap();
        assert_eq!(from_ns(to_ns(dt)).unwrap(), dt);
        assert_eq!(format_ns(to_ns(dt), false), "2000-01-02T03:04:05.500000000");
        let before = parse_datetime("1960-06-01").unwrap();
        assert_eq!(from_ns(to_ns(before)).unwrap(), before);
    }

    #[test]
    fn partial() {
        let (s, e, r) = parse_partial("2000-02").unwrap();
        assert_eq!(r, Resolution::Month);
        assert_eq!(s.to_string(), "2000-02-01 00:00:00");
        assert_eq!(e.to_string(), "2000-03-01 00:00:00");
        assert_eq!(month_end(2000, 1).to_string(), "2000-02-29 00:00:00");
    }
}
