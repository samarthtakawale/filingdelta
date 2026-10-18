from filingdelta.diff import paragraphs, risk_factors


def test_part_header_between_page_fragments_does_not_swallow_the_fragment():
    html = ("<p>ITEM 1A. RISK FACTORS</p><p>Security fails when software updates are</p><p>PART I</p><p>Item 1A</p><p>16</p>"
            "<p>not deployed in time. Other risks follow.</p><p>ITEM 1B. UNRESOLVED</p>")
    assert risk_factors(paragraphs(html)) == ["Security fails when software updates are not deployed in time. Other risks follow."]