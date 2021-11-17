#!/usr/bin/env bash

source .env

if [[ -z "$1" ]]; then
  echo "Name of backup file (without suffix) required!"
  exit 1
fi

DB_CONTAINER_ID=$(docker-compose ps -q db)

if [ -z "$DB_CONTAINER_ID" ] || [ -z "$(docker ps -q --no-trunc | grep $DB_CONTAINER_ID)" ]; then
  echo "Database container is not running!"
  exit 1
fi

BACKUP_COMMANDS="set "PGPASSWORD="$DB_PASSWORD"" && pg_dump -v -U "$DB_USER" -F t "$DB_NAME" > "$DB_BACKUP_DOCKER""$1".tar"

docker exec "$DB_CONTAINER_ID" bash -c "$BACKUP_COMMANDS"