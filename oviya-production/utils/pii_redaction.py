import re


RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"\b(?:\+?\d{1,3})?[\s.-]?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b")
RE_ADDR = re.compile(r"\b\d{1,5}\s+\w+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b", re.IGNORECASE)


def redact(text: str) -> str:
    """
    Redacts email addresses, phone numbers, and street addresses from the given text.
    
    Parameters:
        text (str | None): Input text to redact. If falsy (e.g., None or empty string), the input is returned unchanged.
    
    Returns:
        str: The text with detected emails replaced by "<redacted_email>", phone numbers by "<redacted_phone>", and addresses by "<redacted_address>".
    """
    if not text:
        return text
    t = RE_EMAIL.sub("<redacted_email>", text)
    t = RE_PHONE.sub("<redacted_phone>", t)
    t = RE_ADDR.sub("<redacted_address>", t)
    return t

