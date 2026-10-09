-- Crossing events sent by `detect_bikes.py --report`. Run once in the Supabase SQL editor.
-- Already ran an earlier version? `drop table public.crossings;` first.
--
-- One row per counted crossing, kept small because the board runs 24/7. Anyone with the
-- publishable key can read (the dashboard). Only Auth users whose app_metadata says
-- {"bikecount_role": "uploader"} can insert; nobody can update or delete through the API. app_metadata can only be set with admin rights (SQL editor, secret key),
-- so a stranger who signs up cannot make themselves an uploader.

create table if not exists public.crossings (
  event_id     uuid        primary key,              -- generated on the board; retries dedup on it
  occurred_at  timestamptz not null,                 -- board's wall clock at the crossing
  class_name   text        not null,                 -- bicycle | car | pedestrian | rider(person)
  direction    text        not null check (direction in ('in', 'out'))
);

create index if not exists crossings_occurred_at_idx on public.crossings (occurred_at);

alter table public.crossings enable row level security;

revoke all on public.crossings from anon, authenticated;
grant select, insert on public.crossings to authenticated;
grant select on public.crossings to anon;

create policy "uploaders insert"
  on public.crossings for insert to authenticated
  with check ((auth.jwt() -> 'app_metadata' ->> 'bikecount_role') = 'uploader');

-- Counts are public: the dashboard reads them with just the publishable key. Signed-in users
-- (the uploader reading back what it sent) are included.
create policy "everyone reads"
  on public.crossings for select to anon, authenticated
  using (true);

-- After creating the board's user under Authentication -> Users, flag it as an uploader.
-- Replace the email. Takes effect at its next sign-in.
update auth.users
   set raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb) || '{"bikecount_role": "uploader"}'
 where email = 'REPLACE-WITH-DEVICE-EMAIL';
