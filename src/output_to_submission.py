from pathlib import Path
from datetime import datetime
import pandas as pd

def make_submissions():
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "output"
    submission_dir = output_dir / "submission"

    submission_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        print("output 폴더에 csv 파일이 없습니다.")
        return

    for csv_file in csv_files:
        try:
            mtime = csv_file.stat().st_mtime
            file_time = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M")

            file_name = csv_file.stem.split(".")[0]

            df = pd.read_csv(csv_file)

            if {"ID", "target"}.issubset(df.columns):
                sub_df = df[["ID", "target"]]

                base_save_name = f"{file_time}_{file_name}.csv"
                save_path = submission_dir / base_save_name

                counter = 1
                while save_path.exists():
                    save_path = submission_dir / f"{file_time}_{file_name}_{counter}.csv"
                    counter += 1

                sub_df.to_csv(save_path, index=False)
                print(f"저장 완료: {save_path}")
            else:
                print(f"[Error] ID, target 컬럼을 확인하세요: {csv_file}")
        except Exception as e:
            print(f"[Error] {csv_file} 변환 실패: {e}")


if __name__ == "__main__":
    make_submissions()
