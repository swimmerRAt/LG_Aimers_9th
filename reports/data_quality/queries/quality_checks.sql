WITH quality_checks(check_name, dataset, failed_rows, affected_rate, severity, interpretation) AS (
    VALUES
        ('Train row_id duplicates', 'train.csv', 0, 0.0, 'Pass', '투구 단위 키가 고유함'),
        ('Train domain violations', 'train.csv', 0, 0.0, 'Pass', '타깃·카운트·주자·점수·비율 규칙 위반 없음'),
        ('Train/test feature schema mismatch', 'train.csv + test.csv', 0, 0.0, 'Pass', '47개 평가 피처와 순서가 일치함'),
        ('Trackman ID duplicates', 'trackman_history.csv', 0, 0.0, 'Pass', 'Trackman 행 키가 고유함'),
        ('Trackman count exceptions', 'trackman_history.csv', 97, 0.00005409692160631049, 'Low', '파생 집계 전 제외 권장'),
        ('Direct pitcher ID overlap', 'train.csv + trackman_history.csv', 0, NULL, 'High', '공식 매핑 없이는 선수 단위 직접 조인 금지')
)
SELECT *
FROM quality_checks
ORDER BY failed_rows DESC, check_name;

