# backend — поручения, хранение, экспорт

API:
- POST  /api/meetings              загрузка записи -> {"id": "m1"}
- GET   /api/meetings/{id}         {status, transcript, tasks, summary}
- PATCH /api/tasks/{id}            правка ответственного, срока, статуса
- GET   /api/meetings/{id}/export  протокол в DOCX

Поручение:
```json
{"id":1,"text":"подготовить претензию поставщику","assignee":"Ерлан",
 "deadline_raw":"до конца недели","deadline":"2026-09-26",
 "status":"в работе","quote":"пусть Ерлан до конца недели подготовит претензию"}
```
