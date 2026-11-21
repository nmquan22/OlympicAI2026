"""Render the 2026 workshop handbook from its maintained JSON registry."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    registry = json.loads((ROOT / "catalog/workshops_2026.json").read_text(encoding="utf-8"))
    entries = registry["entries"]
    index = ["# Workshop và shared task 2026 để luyện Olympic AI", "",
             f"Đối chiếu nguồn ngày **{registry['checked_on']}**: **{len(entries)} hướng luyện từ {len(set(e['event'] for e in entries))} sự kiện**.", "",
             "Đây là danh mục chọn lọc theo độ hữu ích cho bài thi 5 tiếng. A = nên luyện trước, B = mở rộng độ phủ, C = khám phá khi bài cấp đủ dữ liệu/weights. Nhãn này là đánh giá của người biên soạn, không phải xác suất ra đề hay khuyến nghị chính thức của IOAI.", "",
             "Cách chọn ưu tiên: đối chiếu kỹ năng với đề IOAI đã công bố/syllabus đúng năm; khả năng tạo baseline bằng model cơ bản hoặc feature được cấp; tốc độ vòng lặp train–dev; và mức bổ sung cho các mảng ít được luyện. Bài cần simulation, spatial audio hoặc generative weights lớn thường ở C. Chưa có dữ liệu thống kê về tần suất đề Olympic AI Việt Nam để gán phần trăm.", "",
             "Nhiều dòng là các track của cùng workshop. Một số sự kiện là challenge/conference, đã ghi rõ trên từng thẻ. **39 hướng không có nghĩa là 39 workshop mới thành lập**. Chỉ gắn nhãn lần đầu khi nguồn xác nhận.", "",
             "Mỗi thẻ tách sự kiện đã xác minh, đề luyện tự biên soạn, baseline, nâng cấp và hướng đọc thêm. Metric bài luyện không thay scorer challenge. Code liên kết là khối nền hoặc baseline rút gọn; chưa tái lập giải pháp thắng/SOTA 2026.", "",
             "Xem thêm [đề IOAI 2026](IOAI_2026.md), [bản đồ khối code](BLOCKS.md), [kế hoạch luyện](PRACTICE_PLAN.md).", ""]
    for area in ["ml", "nlp", "vision", "vision_ml", "audio"]:
        index += [f"## {area}", "", "| Mức | Hướng luyện | Sự kiện |", "|---|---|---|"]
        for e in (e for e in entries if e["area"] == area):
            index.append(f"| {e['priority']} | [{e['title']}](workshops/{e['id']}.md) | {e['event']} |")
        index.append("")
    index += ["## Dùng danh mục offline", "", "```bash", "python -m catalog.search --area ml --priority A", "python -m catalog.search --query retrieval", "python tools/render_catalog.py", "```", "",
              "Dữ liệu gốc nằm ở [workshops_2026.json](../catalog/workshops_2026.json). Sửa registry rồi render lại để bảng và thẻ không lệch nhau. Các URL không được gọi tự động trong lúc chạy baseline.", ""]
    (ROOT / "docs/WORKSHOPS_2026.md").write_text("\n".join(index), encoding="utf-8")
    for e in entries:
        first = "Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này." if e["first_edition"] else "Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập."
        block_links = "\n".join(f"- [{p}](../../{p})" for p in e["blocks"])
        card = f"""# {e['title']}

**{e['event']} · {e['kind']} · ưu tiên {e['priority']} · nguồn kiểm tra {e['checked_on']}.**

## Sự kiện đã xác minh

{e['source_fact']} {first} [Nguồn chính thức]({e['source_url']}).

## Đề luyện do repo đề xuất

{e['practice_io']}

Metric để luyện: **{e['practice_metric']}** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | {e['baseline']} |
| Nâng cấp vừa sức | {e['upgrade']} |
| Hướng nghiên cứu để đọc tiếp | {e['research_frontier']} |

Phép thử quyết định bước tiếp theo: **{e['decisive_experiment']}**

## Chuẩn bị và khối code

{e['requirements']}

{block_links}

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
"""
        (ROOT / f"docs/workshops/{e['id']}.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    render()
